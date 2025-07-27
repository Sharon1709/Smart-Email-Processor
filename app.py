import streamlit as st
import asyncio
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Optional, Any, Literal, TypedDict
from dataclasses import dataclass
from enum import Enum
import logging

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Gmail API imports
import pickle
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure enhanced logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a separate logger for workflow tracking
workflow_logger = logging.getLogger('EmailWorkflow')
workflow_logger.setLevel(logging.INFO)

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/gmail.send']

# Enums and Type Definitions
class EmailCategory(str, Enum):
    SUPPORT = "support"
    SALES = "sales"
    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    INTERNAL = "internal"
    VENDOR = "vendor"
    SPAM = "spam"

class SenderRelationship(str, Enum):
    CUSTOMER = "customer"
    VENDOR = "vendor"
    INTERNAL = "internal"
    UNKNOWN = "unknown"

class UrgencyLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class IntentType(str, Enum):
    QUESTION = "question"
    COMPLAINT = "complaint"
    REQUEST = "request"
    INFORMATION = "information"
    ESCALATION = "escalation"

class RoutingDecision(str, Enum):
    AUTO_RESPONSE = "auto_response"
    HUMAN_ESCALATION = "human_escalation"
    SALES_ROUTING = "sales_routing"
    SUPPORT_ROUTING = "support_routing"
    INTERNAL_ROUTING = "internal_routing"

# Pydantic Models for structured output
class EmailClassification(BaseModel):
    category: EmailCategory = Field(description="Email category")
    sender_relationship: SenderRelationship = Field(description="Sender relationship")
    urgency_level: UrgencyLevel = Field(description="Urgency level")
    key_entities: List[str] = Field(description="Key entities extracted")
    confidence_score: float = Field(description="Classification confidence")

class IntentAnalysis(BaseModel):
    primary_intent: IntentType = Field(description="Primary intent")
    action_required: str = Field(description="Action required")
    complexity_level: Literal["simple", "medium", "complex"] = Field(description="Complexity level")
    key_information: List[str] = Field(description="Key information requirements")

class RoutingDecisionOutput(BaseModel):
    decision: RoutingDecision = Field(description="Routing decision")
    destination: str = Field(description="Destination team/department")
    reasoning: str = Field(description="Reasoning for decision")

# Enhanced State Definition with logging
class EmailProcessingState(TypedDict):
    raw_email: Dict[str, Any]
    structured_email: Dict[str, Any]
    classification: Dict[str, Any]
    context_data: Dict[str, Any]
    intent_analysis: Dict[str, Any]
    routing_decision: Dict[str, Any]
    response_template: Optional[str]
    generated_response: Optional[str]
    escalation_package: Optional[Dict[str, Any]]
    processing_status: str
    error_message: Optional[str]
    processing_history: List[Dict[str, Any]]
    current_node: str
    node_start_time: str
    processing_logs: List[str]
    forwarded_to: Optional[str]

# Utility functions
def safe_str(text: Any) -> str:
    """Safely convert any input to string with proper Unicode handling"""
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    try:
        return str(text)
    except Exception as e:
        logger.warning(f"Error converting to string: {e}")
        return repr(text)

def clean_text_for_processing(text: str) -> str:
    """Clean text for LLM processing while preserving Unicode characters"""
    if not text:
        return ""
    try:
        cleaned = text.replace('\x00', '').replace('\r', '\n')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        logger.warning(f"Error cleaning text: {e}")
        return text

def extract_email_address(sender_string: str) -> str:
    """Extract clean email address from sender string"""
    if not sender_string:
        return ""
    if '<' in sender_string and '>' in sender_string:
        start = sender_string.find('<')
        end = sender_string.find('>')
        if start != -1 and end != -1 and end > start:
            return sender_string[start + 1:end].strip()
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, sender_string)
    if matches:
        return matches[0]
    return sender_string.strip()

def log_node_entry(state: EmailProcessingState, node_name: str) -> EmailProcessingState:
    """Log entry to a workflow node"""
    current_time = datetime.now().isoformat()
    workflow_logger.info(f"🔄 ENTERING NODE: {node_name} at {current_time}")
    updated_state = dict(state)
    updated_state["current_node"] = node_name
    updated_state["node_start_time"] = current_time
    if "processing_logs" not in updated_state:
        updated_state["processing_logs"] = []
    updated_state["processing_logs"].append(f"ENTER {node_name}: {current_time}")
    return updated_state

def log_node_exit(state: EmailProcessingState, node_name: str, status: str = "success") -> EmailProcessingState:
    """Log exit from a workflow node"""
    current_time = datetime.now().isoformat()
    start_time = state.get("node_start_time", current_time)
    workflow_logger.info(f"✅ EXITING NODE: {node_name} at {current_time} - Status: {status}")
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(current_time)
        duration = (end_dt - start_dt).total_seconds()
        workflow_logger.info(f"⏱️ NODE DURATION: {node_name} took {duration:.2f} seconds")
    except Exception as e:
        workflow_logger.warning(f"Could not calculate duration for {node_name}: {e}")
        duration = 0
    updated_state = dict(state)
    history_entry = {
        "node": node_name,
        "timestamp": current_time,
        "status": status,
        "duration_seconds": duration
    }
    if "processing_history" not in updated_state:
        updated_state["processing_history"] = []
    updated_state["processing_history"].append(history_entry)
    updated_state["processing_logs"].append(f"EXIT {node_name}: {current_time} - {status} ({duration:.2f}s)")
    return updated_state

# Gmail Service Class
class GmailService:
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.pickle"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.authenticated = False

    def _authenticate(self):
        """Authenticate with Gmail API"""
        try:
            logger.info("🔐 Starting Gmail authentication...")
            creds = None
            if os.path.exists(self.token_path):
                try:
                    with open(self.token_path, 'rb') as token:
                        creds = pickle.load(token)
                except Exception as e:
                    logger.warning(f"⚠️ Error loading token: {e}")
                    if os.path.exists(self.token_path):
                        os.remove(self.token_path)
                    creds = None
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        logger.error(f"❌ Error refreshing token: {e}")
                        if os.path.exists(self.token_path):
                            os.remove(self.token_path)
                        creds = None
                if not creds:
                    if not os.path.exists(self.credentials_path):
                        raise FileNotFoundError(f"Gmail credentials file not found at {self.credentials_path}")
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)
            self.service = build('gmail', 'v1', credentials=creds)
            self.authenticated = True
            logger.info("✅ Gmail service initialized")
        except Exception as e:
            logger.error(f"❌ Gmail authentication failed: {e}")
            self.authenticated = False
            raise e

    def ensure_authenticated(self):
        """Ensure Gmail service is authenticated"""
        if not self.authenticated or not self.service:
            self._authenticate()

    def get_recent_emails(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get recent emails from Gmail"""
        try:
            self.ensure_authenticated()
            results = self.service.users().messages().list(userId='me', maxResults=max_results).execute()
            messages = results.get('messages', [])
            emails = []
            for message in messages:
                msg = self.service.users().messages().get(userId='me', id=message['id']).execute()
                parsed_email = self._parse_email(msg)
                emails.append(parsed_email)
            return emails
        except Exception as e:
            logger.error(f"❌ Error fetching emails: {e}")
            return []

    def _parse_email(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Gmail message into structured format"""
        try:
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            body = self._extract_body(payload)
            return {
                'id': msg['id'],
                'subject': subject,
                'sender': sender,
                'sender_email': extract_email_address(sender),
                'date': date,
                'body': body,
                'thread_id': msg.get('threadId'),
                'labels': msg.get('labelIds', []),
                'snippet': msg.get('snippet', ''),
                'raw_data': msg
            }
        except Exception as e:
            logger.error(f"❌ Error parsing email: {e}")
            return {'id': msg.get('id', 'unknown'), 'subject': 'Error', 'sender': 'unknown', 'sender_email': '', 'date': '', 'body': '', 'thread_id': '', 'labels': [], 'snippet': '', 'raw_data': msg}

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract email body from payload"""
        try:
            body = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('mimeType') == 'text/plain':
                        data = part.get('body', {}).get('data', '')
                        if data:
                            decoded_bytes = base64.urlsafe_b64decode(data)
                            body = decoded_bytes.decode('utf-8', errors='replace')
                            break
            else:
                if payload.get('mimeType') == 'text/plain':
                    data = payload.get('body', {}).get('data', '')
                    if data:
                        decoded_bytes = base64.urlsafe_b64decode(data)
                        body = decoded_bytes.decode('utf-8', errors='replace')
            return clean_text_for_processing(body)
        except Exception as e:
            logger.warning(f"⚠️ Error extracting email body: {e}")
            return "Error extracting email body"

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email via Gmail API"""
        try:
            logger.info(f"📤 Attempting to send email to: {to}")
            self.ensure_authenticated()
            clean_to = extract_email_address(to)
            if not clean_to:
                logger.error(f"❌ Invalid email address: {to}")
                return False
            message = MIMEText(body, _charset='utf-8')
            message['to'] = clean_to
            message['subject'] = subject
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            self.service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
            logger.info(f"✅ Email sent successfully to: {clean_to}")
            return True
        except Exception as e:
            logger.error(f"❌ Error sending email to {to}: {e}")
            return False

# Email Processing Agent
class EmailProcessingAgent:
    def __init__(self, groq_api_key: str, model_name: str = "llama3-70b-8192"):
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.1)
        self.gmail_service = GmailService()

    def intake_node(self, state: EmailProcessingState) -> EmailProcessingState:
        """Email Intake Node"""
        state = log_node_entry(state, "intake")
        try:
            raw_email = state["raw_email"]
            structured_email = {
                "id": safe_str(raw_email.get("id")),
                "subject": clean_text_for_processing(safe_str(raw_email.get("subject", ""))),
                "sender": safe_str(raw_email.get("sender", "")),
                "sender_email": safe_str(raw_email.get("sender_email", "")),
                "date": safe_str(raw_email.get("date", "")),
                "body": clean_text_for_processing(safe_str(raw_email.get("body", ""))),
                "thread_id": safe_str(raw_email.get("thread_id")),
                "labels": raw_email.get("labels", []),
                "snippet": clean_text_for_processing(safe_str(raw_email.get("snippet", ""))),
                "attachments": [],
                "word_count": len(safe_str(raw_email.get("body", "")).split()),
                "has_attachments": False,
                "processed_at": datetime.now().isoformat()
            }
            state = log_node_exit(state, "intake", "success")
            return {**state, "structured_email": structured_email, "processing_status": "intake_complete"}
        except Exception as e:
            state = log_node_exit(state, "intake", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def classification_agent(self, state: EmailProcessingState) -> EmailProcessingState:
        """Classification Agent"""
        state = log_node_entry(state, "classification")
        try:
            structured_email = state["structured_email"]
            classification_prompt = ChatPromptTemplate.from_messages([
                ("system", """Classify email into: support, sales, complaint, inquiry, internal, vendor, spam.
Sender Relationships: customer, vendor, internal, unknown.
Urgency Levels: high, medium, low.
Return JSON with category, sender_relationship, urgency_level, key_entities, confidence_score."""),
                ("human", "Subject: {subject}\nSender: {sender}\nBody: {body}")
            ])
            parser = JsonOutputParser(pydantic_object=EmailClassification)
            chain = classification_prompt | self.llm | parser
            classification_result = chain.invoke({
                "subject": safe_str(structured_email.get('subject', '')),
                "sender": safe_str(structured_email.get('sender', '')),
                "body": safe_str(structured_email.get('body', ''))
            })
            state = log_node_exit(state, "classification", "success")
            return {**state, "classification": classification_result, "processing_status": "classification_complete"}
        except Exception as e:
            state = log_node_exit(state, "classification", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def context_enrichment_node(self, state: EmailProcessingState) -> EmailProcessingState:
        """Context Enrichment Node"""
        state = log_node_entry(state, "context_enrichment")
        try:
            structured_email = state["structured_email"]
            context_data = {
                "sender_history": {"previous_emails": 5, "last_interaction": "2024-01-15", "customer_tier": "premium"},
                "enrichment_timestamp": datetime.now().isoformat()
            }
            state = log_node_exit(state, "context_enrichment", "success")
            return {**state, "context_data": context_data, "processing_status": "context_enrichment_complete"}
        except Exception as e:
            state = log_node_exit(state, "context_enrichment", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def intent_analysis_agent(self, state: EmailProcessingState) -> EmailProcessingState:
        """Intent Analysis Agent"""
        state = log_node_entry(state, "intent_analysis")
        try:
            structured_email = state["structured_email"]
            classification = state["classification"]
            intent_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze intent: question, complaint, request, information, escalation.
Return JSON with primary_intent, action_required, complexity_level, key_information."""),
                ("human", "Subject: {subject}\nBody: {body}\nClassification: {classification}")
            ])
            parser = JsonOutputParser(pydantic_object=IntentAnalysis)
            chain = intent_prompt | self.llm | parser
            intent_result = chain.invoke({
                "subject": safe_str(structured_email.get('subject', '')),
                "body": safe_str(structured_email.get('body', '')),
                "classification": json.dumps(classification)
            })
            state = log_node_exit(state, "intent_analysis", "success")
            return {**state, "intent_analysis": intent_result, "processing_status": "intent_analysis_complete"}
        except Exception as e:
            state = log_node_exit(state, "intent_analysis", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def routing_decision_node(self, state: EmailProcessingState) -> EmailProcessingState:
        """Routing Decision Node"""
        state = log_node_entry(state, "routing_decision")
        try:
            classification = state["classification"]
            intent_analysis = state["intent_analysis"]
            
            # Fixed prompt without the problematic variables
            routing_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a routing decision agent. Based on the email classification and intent analysis, determine the best routing decision.

Routing Options:
- auto_response: Simple queries that can be handled automatically
- human_escalation: Complex issues requiring human intervention
- sales_routing: Sales-related inquiries
- support_routing: Technical support issues
- internal_routing: Internal communications

Decision Criteria:
- Simple FAQ or common questions -> auto_response
- Complex technical issues -> support_routing
- Sales inquiries -> sales_routing
- Complaints or escalations -> human_escalation
- Internal communications -> internal_routing

CRITICAL: You must respond with ONLY a valid JSON object. No explanatory text before or after. The JSON must have exactly these keys:
- decision: one of the routing options above
- destination: the team name
- reasoning: your explanation

Example response (copy this format exactly):
{{"decision": "support_routing", "destination": "Support Team", "reasoning": "Technical issue reported requiring expert assistance"}}

Do not include any other text. Start your response with {{ and end with }}."""),
                ("human", """Classification: {classification}
Intent Analysis: {intent_analysis}

Respond with JSON only:""")
            ])
            
            chain = routing_prompt | self.llm
            
            classification_str = json.dumps(classification) if isinstance(classification, dict) else safe_str(classification)
            intent_str = json.dumps(intent_analysis) if isinstance(intent_analysis, dict) else safe_str(intent_analysis)
            
            workflow_logger.info("🤖 Sending data to LLM for routing decision...")
            
            try:
                # Get the raw response from LLM
                raw_response = chain.invoke({
                    "classification": classification_str,
                    "intent_analysis": intent_str
                })
                
                # Extract content if it's a message object
                if hasattr(raw_response, 'content'):
                    response_text = raw_response.content
                else:
                    response_text = str(raw_response)
                
                workflow_logger.info(f"🔍 Raw LLM response: {response_text}")
                
                # Clean the response - extract JSON from any surrounding text
                response_text = response_text.strip()
                
                # Try to find JSON within the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    workflow_logger.info(f"🔧 Extracted JSON: {json_text}")
                    
                    # Parse the JSON
                    routing_result = json.loads(json_text)
                    
                    # Validate the required keys
                    required_keys = ['decision', 'destination', 'reasoning']
                    if not all(key in routing_result for key in required_keys):
                        raise ValueError(f"Missing required keys. Expected: {required_keys}, Got: {list(routing_result.keys())}")
                    
                    # Validate decision is one of the allowed values
                    valid_decisions = ["auto_response", "human_escalation", "sales_routing", "support_routing", "internal_routing"]
                    if routing_result['decision'] not in valid_decisions:
                        workflow_logger.warning(f"⚠️ Invalid decision '{routing_result['decision']}', defaulting to human_escalation")
                        routing_result['decision'] = "human_escalation"
                        routing_result['destination'] = "Management Team"
                        routing_result['reasoning'] = f"Invalid routing decision received: {routing_result['decision']}. Defaulted to human escalation."
                    
                    workflow_logger.info(f"✅ Valid routing result: {routing_result}")
                    
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except json.JSONDecodeError as e:
                workflow_logger.error(f"❌ JSON parsing failed: {e}")
                workflow_logger.error(f"❌ Problematic response: {response_text}")
                raise ValueError(f"Invalid JSON format: {str(e)}")
                
            except Exception as e:
                workflow_logger.error(f"❌ LLM processing failed: {e}")
                raise e
            
            state = log_node_exit(state, "routing_decision", "success")
            return {
                **state,
                "routing_decision": routing_result,
                "processing_status": "routing_decision_complete"
            }
            
        except Exception as e:
            workflow_logger.error(f"❌ Error in routing decision node: {e}")
            state = log_node_exit(state, "routing_decision", "error")
            
            # Provide a robust fallback
            fallback_routing = {
                "decision": "human_escalation",
                "destination": "Management Team",
                "reasoning": f"Error in routing decision: {str(e)}. Defaulting to human escalation for safety."
            }
            
            return {
                **state,
                "routing_decision": fallback_routing,
                "processing_status": "routing_decision_error",
                "error_message": str(e)
            }

    def email_forwarding_node(self, state: EmailProcessingState) -> EmailProcessingState:
        """Email Forwarding Node - Forward email to the appropriate team"""
        state = log_node_entry(state, "email_forwarding")
        try:
            workflow_logger.info("📤 Initiating email forwarding...")
            structured_email = state["structured_email"]
            routing_decision = state.get("routing_decision", {})
            import streamlit as st
            default_email = "endurisharon181@gmail.com"
            team_emails = {
                "support_routing": st.session_state.get('support_team', default_email),
                "sales_routing": st.session_state.get('sales_team', default_email),
                "internal_routing": st.session_state.get('management_team', default_email),
                "human_escalation": st.session_state.get('management_team', default_email),
                "auto_response": None
            }
            decision = routing_decision.get("decision", "human_escalation") if isinstance(routing_decision, dict) else "human_escalation"
            destination_email = team_emails.get(decision, default_email)
            workflow_logger.info(f"🛠️ Routing decision: {decision}")
            workflow_logger.info(f"📧 Destination email: {destination_email}")
            if destination_email:
                subject = f"Fwd: {safe_str(structured_email.get('subject', 'No Subject'))}"
                body = f"""
Forwarded Email:
----------------
From: {safe_str(structured_email.get('sender', 'Unknown'))}
Subject: {safe_str(structured_email.get('subject', 'No Subject'))}
Date: {safe_str(structured_email.get('date', 'Unknown'))}
Category: {safe_str(state.get('classification', {}).get('category', 'Unknown'))}
Routing Decision: {safe_str(decision)}
Reasoning: {safe_str(routing_decision.get('reasoning', 'Unknown') if isinstance(routing_decision, dict) else 'Invalid routing decision')}
----------------
Original Message:
{safe_str(structured_email.get('body', ''))}
"""
                workflow_logger.info(f"📧 Preparing to forward to: {destination_email}")
                success = self.gmail_service.send_email(destination_email, subject, body)
                if success:
                    workflow_logger.info(f"✅ Email forwarded successfully to {destination_email}")
                    state["forwarded_to"] = destination_email
                else:
                    workflow_logger.error(f"❌ Failed to forward email to {destination_email}")
                    state["error_message"] = f"Failed to forward email to {destination_email}"
                    state["processing_status"] = "forwarding_error"
                    state = log_node_exit(state, "email_forwarding", "error")
                    return state
            else:
                workflow_logger.info("ℹ️ No forwarding needed for auto_response")
            state = log_node_exit(state, "email_forwarding", "success")
            return {**state, "processing_status": "email_forwarded"}
        except Exception as e:
            workflow_logger.error(f"❌ Error in email forwarding node: {e}")
            state = log_node_exit(state, "email_forwarding", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def auto_response_generation_agent(self, state: EmailProcessingState) -> EmailProcessingState:
        """Auto-Response Generation Agent"""
        state = log_node_entry(state, "auto_response_generation")
        try:
            structured_email = state["structured_email"]
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """Generate a professional email response. Start with a greeting, address the email content, and end with a signature."""),
                ("human", "Subject: {subject}\nBody: {body}\nSender: {sender}")
            ])
            chain = response_prompt | self.llm
            generated_response = chain.invoke({
                "subject": safe_str(structured_email.get('subject', '')),
                "body": safe_str(structured_email.get('body', '')),
                "sender": safe_str(structured_email.get('sender', ''))
            }).content
            state = log_node_exit(state, "auto_response_generation", "success")
            return {**state, "generated_response": generated_response, "processing_status": "auto_response_generated"}
        except Exception as e:
            state = log_node_exit(state, "auto_response_generation", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def escalation_preparation_agent(self, state: EmailProcessingState) -> EmailProcessingState:
        """Escalation Preparation Agent"""
        state = log_node_entry(state, "escalation_preparation")
        try:
            structured_email = state["structured_email"]
            classification = state["classification"]
            routing_decision = state["routing_decision"]
            escalation_package = {
                "summary": f"Email from {safe_str(structured_email.get('sender', ''))}",
                "priority": classification.get("urgency_level", "medium"),
                "destination_team": safe_str(routing_decision.get("destination", "") if isinstance(routing_decision, dict) else "Unknown")
            }
            state = log_node_exit(state, "escalation_preparation", "success")
            return {**state, "escalation_package": escalation_package, "processing_status": "escalation_prepared"}
        except Exception as e:
            state = log_node_exit(state, "escalation_preparation", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def quality_assurance_node(self, state: EmailProcessingState) -> EmailProcessingState:
        """Quality Assurance Node"""
        state = log_node_entry(state, "quality_assurance")
        try:
            generated_response = safe_str(state.get("generated_response", ""))
            quality_score = 0.9  # Simulated
            state = log_node_exit(state, "quality_assurance", "success")
            return {**state, "quality_score": quality_score, "processing_status": "quality_assured"}
        except Exception as e:
            state = log_node_exit(state, "quality_assurance", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

    def learning_optimization_node(self, state: EmailProcessingState) -> EmailProcessingState:
        """Learning and Optimization Node"""
        state = log_node_entry(state, "learning_optimization")
        try:
            metrics = {"processing_time": sum(step.get("duration_seconds", 0) for step in state.get("processing_history", []))}
            state = log_node_exit(state, "learning_optimization", "success")
            return {**state, "performance_metrics": metrics, "processing_status": "learning_complete"}
        except Exception as e:
            state = log_node_exit(state, "learning_optimization", "error")
            return {**state, "processing_status": "error", "error_message": str(e)}

# Workflow Definition
def create_email_processing_workflow(agent: EmailProcessingAgent):
    """Create the LangGraph workflow"""
    workflow = StateGraph(EmailProcessingState)
    workflow.add_node("intake", agent.intake_node)
    workflow.add_node("classification", agent.classification_agent)
    workflow.add_node("context_enrichment", agent.context_enrichment_node)
    workflow.add_node("intent_analysis", agent.intent_analysis_agent)
    workflow.add_node("routing_decision", agent.routing_decision_node)
    workflow.add_node("email_forwarding", agent.email_forwarding_node)
    workflow.add_node("auto_response_generation", agent.auto_response_generation_agent)
    workflow.add_node("escalation_preparation", agent.escalation_preparation_agent)
    workflow.add_node("quality_assurance", agent.quality_assurance_node)
    workflow.add_node("learning_optimization", agent.learning_optimization_node)
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "classification")
    workflow.add_edge("classification", "context_enrichment")
    workflow.add_edge("context_enrichment", "intent_analysis")
    workflow.add_edge("intent_analysis", "routing_decision")
    workflow.add_edge("routing_decision", "email_forwarding")
    def route_decision(state: EmailProcessingState):
        decision = state.get("routing_decision", {}).get("decision", "auto_response") if isinstance(state.get("routing_decision"), dict) else "auto_response"
        return "auto_response_generation" if decision == "auto_response" else "escalation_preparation"
    workflow.add_conditional_edges("email_forwarding", route_decision, {
        "auto_response_generation": "auto_response_generation",
        "escalation_preparation": "escalation_preparation"
    })
    workflow.add_edge("auto_response_generation", "quality_assurance")
    workflow.add_edge("quality_assurance", "learning_optimization")
    workflow.add_edge("escalation_preparation", "learning_optimization")
    workflow.add_edge("learning_optimization", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Streamlit Application
def main():
    st.set_page_config(page_title="Smart Email Processor", page_icon="📧", layout="wide")
    st.title("🤖 Smart Email Processor")
    st.sidebar.header("Configuration")
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    model_options = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    st.sidebar.header("Gmail Setup")
    creds_file_exists = os.path.exists("credentials.json")
    token_file_exists = os.path.exists("token.pickle")
    if creds_file_exists:
        st.sidebar.success("✅ Credentials file found")
    else:
        st.sidebar.error("❌ credentials.json missing")
    if token_file_exists:
        st.sidebar.info("🔄 Token file exists")
    else:
        st.sidebar.warning("⚠️ No saved token")
    if st.sidebar.button("🔄 Reset Gmail Authentication"):
        if os.path.exists("token.pickle"):
            os.remove("token.pickle")
            st.sidebar.success("Token cleared! Please re-authenticate.")
            st.rerun()
    if not groq_api_key:
        st.warning("Please enter your Groq API key")
        st.stop()
    @st.cache_resource
    def get_agent_and_workflow(api_key, model):
        agent = EmailProcessingAgent(api_key, model)
        workflow = create_email_processing_workflow(agent)
        return agent, workflow
    try:
        agent, workflow = get_agent_and_workflow(groq_api_key, selected_model)
        st.sidebar.success("✅ Agent initialized")
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        st.stop()
    if 'emails' not in st.session_state:
        st.session_state['emails'] = []
    if 'processing_result' not in st.session_state:
        st.session_state['processing_result'] = None
    if 'selected_email_index' not in st.session_state:
        st.session_state['selected_email_index'] = 0
    if 'current_selected_email' not in st.session_state:
        st.session_state['current_selected_email'] = None
    tab1, tab2, tab3 = st.tabs(["📧 Process Emails", "📊 Dashboard", "⚙️ Settings"])
    with tab1:
        st.header("Email Processing")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Fetch Recent Emails")
            max_emails = st.slider("Number of emails to fetch", 1, 20, 5)
            if st.button("🔄 Fetch Emails", type="primary"):
                try:
                    with st.spinner("Authenticating with Gmail..."):
                        agent.gmail_service.ensure_authenticated()
                    with st.spinner("Fetching emails..."):
                        emails = agent.gmail_service.get_recent_emails(max_emails)
                    if emails:
                        st.success(f"Fetched {len(emails)} emails")
                        st.session_state['emails'] = emails
                        st.session_state['processing_result'] = None
                        st.session_state['current_selected_email'] = None
                        email_data = [
                            {
                                'Subject': safe_str(email.get('subject', ''))[:50] + '...' if len(safe_str(email.get('subject', ''))) > 50 else safe_str(email.get('subject', '')),
                                'From': safe_str(email.get('sender', '')),
                                'Date': safe_str(email.get('date', '')),
                                'Snippet': safe_str(email.get('snippet', ''))[:100] + '...' if len(safe_str(email.get('snippet', ''))) > 100 else safe_str(email.get('snippet', ''))
                            } for email in emails
                        ]
                        st.dataframe(email_data, use_container_width=True)
                    else:
                        st.warning("No emails found")
                except Exception as e:
                    st.error(f"Error fetching emails: {e}")
        with col2:
            st.subheader("Process Selected Email")
            if st.session_state['emails']:
                email_options = [f"{safe_str(email.get('subject', 'No Subject'))} - {safe_str(email.get('sender', 'Unknown'))}" for email in st.session_state['emails']]
                selected_idx = st.selectbox("Select email to process", range(len(email_options)), format_func=lambda x: email_options[x], key="email_selector")
                if selected_idx != st.session_state.get('selected_email_index', 0):
                    st.session_state['selected_email_index'] = selected_idx
                    st.session_state['current_selected_email'] = st.session_state['emails'][selected_idx]
                    st.session_state['processing_result'] = None
                elif st.session_state.get('current_selected_email') is None:
                    st.session_state['current_selected_email'] = st.session_state['emails'][selected_idx]
                if st.button("🚀 Process Email", type="primary"):
                    selected_email = st.session_state['emails'][selected_idx]
                    st.session_state['current_selected_email'] = selected_email
                    with st.spinner("Processing email..."):
                        initial_state = {
                            "raw_email": selected_email,
                            "structured_email": {},
                            "classification": {},
                            "context_data": {},
                            "intent_analysis": {},
                            "routing_decision": {},
                            "response_template": None,
                            "generated_response": None,
                            "escalation_package": None,
                            "processing_status": "initialized",
                            "error_message": None,
                            "processing_history": [],
                            "processing_logs": [],
                            "forwarded_to": None
                        }
                        try:
                            config = {"configurable": {"thread_id": safe_str(selected_email.get('id', 'unknown'))}}
                            result = workflow.invoke(initial_state, config)
                            st.session_state['processing_result'] = result
                            if result.get('processing_status') == 'error' or result.get('processing_status') == 'routing_decision_error':
                                st.error(f"Processing failed: {result.get('error_message', 'Unknown error')}")
                            else:
                                st.success("Email processed successfully!")
                                if result.get('forwarded_to'):
                                    st.success(f"Email forwarded to: {result['forwarded_to']}")
                                elif result.get('processing_status') == 'email_forwarded' and not result.get('forwarded_to'):
                                    st.info("No forwarding needed (auto_response)")
                                if result.get('error_message'):
                                    st.warning(f"Processing issue: {result['error_message']}")
                        except Exception as e:
                            st.error(f"Error processing email: {e}")
            else:
                st.info("Please fetch emails first")
            if st.session_state.get('processing_result'):
                result = st.session_state['processing_result']
                st.subheader("Processing Results")
                if 'classification' in result and result['classification']:
                    st.write("**Classification:**")
                    classification = result['classification']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Category", safe_str(classification.get('category', 'Unknown')))
                    with col2:
                        st.metric("Urgency", safe_str(classification.get('urgency_level', 'Unknown')))
                    with col3:
                        st.metric("Confidence", f"{classification.get('confidence_score', 0):.2f}")
                if 'routing_decision' in result and result['routing_decision']:
                    st.write("**Routing Decision:**")
                    routing = result['routing_decision']
                    st.write(f"Decision: {safe_str(routing.get('decision', 'Unknown'))}")
                    st.write(f"Destination: {safe_str(routing.get('destination', 'Unknown'))}")
                    st.write(f"Reasoning: {safe_str(routing.get('reasoning', 'Unknown'))}")
                if result.get('forwarded_to'):
                    st.write(f"**Forwarded To:** {safe_str(result.get('forwarded_to', 'Unknown'))}")
                elif result.get('processing_status') == 'email_forwarded':
                    st.write("**Forwarded To:** Auto-response (no forwarding)")
                if result.get('error_message'):
                    st.warning(f"**Processing Issue:** {result.get('error_message')}")
                if result.get('generated_response'):
                    st.write("**Generated Response:**")
                    response_text = safe_str(result['generated_response'])
                    st.text_area("Auto-generated response", response_text, height=200, key="response_display")
                    if st.button("📤 Send Response", key="send_response_btn"):
                        current_email = st.session_state.get('current_selected_email')
                        if not current_email:
                            st.error("No email selected")
                            st.stop()
                        sender_email = safe_str(current_email.get('sender_email', ''))
                        subject = "Re: " + safe_str(current_email.get('subject', ''))
                        with st.spinner("Sending response..."):
                            if agent.gmail_service.send_email(sender_email, subject, response_text):
                                st.success("Response sent successfully!")
                                st.session_state['last_sent_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                st.error("Failed to send response")
    with tab2:
        st.header("📊 Dashboard")
        st.subheader("Processing Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Processed", "156", "12 today")
        with col2:
            st.metric("Auto-Responses", "89%", "5% increase")
        with col3:
            st.metric("Avg Response Time", "2.3s", "-0.5s")
        if st.session_state.get('processing_result'):
            result = st.session_state['processing_result']
            st.write("**Last Processed Email:**")
            structured_email = result.get('structured_email', {})
            classification = result.get('classification', {})
            st.write(f"Subject: {safe_str(structured_email.get('subject', 'Unknown'))}")
            st.write(f"Category: {safe_str(classification.get('category', 'Unknown'))}")
            st.write(f"Status: {safe_str(result.get('processing_status', 'Unknown'))}")
            if result.get('forwarded_to'):
                st.write(f"Forwarded To: {safe_str(result.get('forwarded_to', 'Unknown'))}")
            elif result.get('processing_status') == 'email_forwarded':
                st.write("Forwarded To: Auto-response (no forwarding)")
            if result.get('error_message'):
                st.write(f"Processing Issue: {safe_str(result.get('error_message'))}")
    with tab3:
        st.header("⚙️ Settings")
        st.subheader("Email Processing Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Auto-Response Settings:**")
            st.checkbox("Enable auto-responses", value=True)
            st.slider("Response delay (seconds)", 0, 300, 30)
        with col2:
            st.write("**Team Assignments:**")
            default_email = "endurisharon181@gmail.com"
            st.session_state['support_team'] = st.text_input("Support Team Email", value=st.session_state.get('support_team', default_email))
            st.session_state['sales_team'] = st.text_input("Sales Team Email", value=st.session_state.get('sales_team', default_email))
            st.session_state['management_team'] = st.text_input("Management Team Email", value=st.session_state.get('management_team', default_email))
        if st.button("💾 Save Settings"):
            st.success("Settings saved!")

if __name__ == "__main__":
    main()