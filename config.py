# config.py - Configuration management
import os
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class ModelConfig:
    """Configuration for different LLM models"""
    name: str
    max_tokens: int
    temperature: float
    recommended_use: str

@dataclass
class TeamConfig:
    """Configuration for team routing"""
    email: str
    escalation_threshold: float
    response_time_sla: int  # in hours
    categories: list

@dataclass
class EmailProcessingConfig:
    """Main configuration class"""
    # API Configuration
    groq_api_key: str
    gmail_credentials_path: str
    
    # Model Configuration
    default_model: str
    available_models: Dict[str, ModelConfig]
    
    # Processing Configuration
    max_emails_per_batch: int
    classification_threshold: float
    auto_response_enabled: bool
    response_delay_seconds: int
    
    # Team Configuration
    teams: Dict[str, TeamConfig]
    
    # Template Configuration
    response_templates: Dict[str, str]
    
    # Escalation Configuration
    escalation_urgency_levels: list
    escalation_categories: list
    
    @classmethod
    def from_env(cls) -> 'EmailProcessingConfig':
        """Load configuration from environment variables"""
        return cls(
            groq_api_key=os.getenv('GROQ_API_KEY', ''),
            gmail_credentials_path=os.getenv('GMAIL_CREDENTIALS_PATH', 'credentials.json'),
            default_model=os.getenv('DEFAULT_MODEL', 'llama3-8b-8192'),
            available_models={
                'llama3-70b-8192': ModelConfig(
                    name='llama3-70b-8192',
                    max_tokens=8192,
                    temperature=0.1,
                    recommended_use='Best quality, complex analysis'
                ),
                'llama3-8b-8192': ModelConfig(
                    name='llama3-8b-8192',
                    max_tokens=8192,
                    temperature=0.1,
                    recommended_use='Balanced performance'
                ),
                'mixtral-8x7b-32768': ModelConfig(
                    name='mixtral-8x7b-32768',
                    max_tokens=32768,
                    temperature=0.1,
                    recommended_use='Fast processing, long context'
                )
            },
            max_emails_per_batch=int(os.getenv('MAX_EMAILS_PER_BATCH', '10')),
            classification_threshold=float(os.getenv('CLASSIFICATION_THRESHOLD', '0.7')),
            auto_response_enabled=os.getenv('AUTO_RESPONSE_ENABLED', 'true').lower() == 'true',
            response_delay_seconds=int(os.getenv('RESPONSE_DELAY_SECONDS', '5')),
            teams={
                'sales': TeamConfig(
                    email='endurisharon181@gmail.com',
                    escalation_threshold=0.8,
                    response_time_sla=2,
                    categories=['sales_inquiry', 'pricing', 'product_demo']
                ),
                'support': TeamConfig(
                    email='endurisharon181@gmail.com',
                    escalation_threshold=0.7,
                    response_time_sla=4,
                    categories=['technical_support', 'bug_report', 'feature_request']
                ),
                'general': TeamConfig(
                    email='endurisharon181@gmail.com',
                    escalation_threshold=0.6,
                    response_time_sla=24,
                    categories=['general_inquiry', 'feedback', 'other']
                )
            },
            response_templates={
                'acknowledgment': """Thank you for your email. We have received your message and will respond within {sla_hours} hours.""",
                'escalation': """Your message has been escalated to our {team} team for specialized attention.""",
                'auto_response': """Thank you for contacting us. Based on your inquiry about {topic}, here's some helpful information: {response}"""
            },
            escalation_urgency_levels=['low', 'medium', 'high', 'urgent'],
            escalation_categories=['complaint', 'legal', 'security', 'urgent_technical']
        )