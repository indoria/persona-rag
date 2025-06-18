{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "journalist_id": {"type": "string"},
    "version": {"type": "integer"},
    "basic_info": {
      "type": "object",
      "properties": {
        "name": {
          "first": {"type": "string"},
          "last": {"type": "string"},
          "display": {"type": "string"}
        },
        "contact": {
          "email": {"type": "string", "format": "email"},
          "phone": {"type": "string"},
          "social_media": {
            "twitter": {"type": "string"},
            "linkedin": {"type": "string"}
          }
        },
        "professional": {
          "title": {"type": "string"},
          "organization": {
            "name": {"type": "string"},
            "department": {"type": "string"},
            "role": {"type": "string"}
          },
          "employment_status": {"enum": ["staff", "freelance", "contract"]}
        }
      }
    },
    "geographic_profile": {
      "location_hierarchy": {
        "country": {"code": {"type": "string"}, "name": {"type": "string"}},
        "region": {"code": {"type": "string"}, "name": {"type": "string"}},
        "city": {"name": {"type": "string"}}
      },
      "coverage_areas": [{
        "type": {"enum": ["primary", "secondary", "occasional"]},
        "geo_bounds": {"type": "object"},
        "expertise_level": {"enum": ["novice", "intermediate", "expert"]}
      }],
      "coordinates": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"}
      }
    },
    "expertise_profile": {
      "primary_beats": [{
        "iptc_code": {"type": "string"},
        "category": {"type": "string"},
        "expertise_level": {"enum": ["novice", "intermediate", "expert"]},
        "years_experience": {"type": "integer"},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
      }],
      "topic_preferences": [{
        "topic": {"type": "string"},
        "interest_level": {"type": "integer", "minimum": 1, "maximum": 10},
        "coverage_frequency": {"type": "string"}
      }]
    },
    "content_analysis": {
      "writing_style": {
        "avg_sentence_length": {"type": "number"},
        "vocabulary_richness": {"type": "number"},
        "formality_score": {"type": "number"},
        "objectivity_score": {"type": "number"}
      },
      "content_patterns": {
        "article_frequency": {"type": "number"},
        "word_count_avg": {"type": "integer"},
        "source_diversity": {"type": "number"},
        "multimedia_usage": {"type": "number"}
      },
      "question_patterns": {
        "interview_style": {"type": "string"},
        "question_complexity": {"type": "number"},
        "follow_up_tendency": {"type": "number"}
      }
    },
    "sentiment_analysis": {
      "overall_sentiment": {
        "polarity": {"type": "number", "minimum": -1, "maximum": 1},
        "subjectivity": {"type": "number", "minimum": 0, "maximum": 1},
        "consistency_score": {"type": "number"}
      },
      "topic_sentiment_map": {
        "type": "object",
        "additionalProperties": {
          "avg_polarity": {"type": "number"},
          "trend": {"enum": ["increasing", "decreasing", "stable"]},
          "sample_size": {"type": "integer"}
        }
      },
      "temporal_patterns": [{
        "time_period": {"type": "string"},
        "sentiment_metrics": {"type": "object"}
      }]
    },
    "crisis_management_profile": {
      "crisis_response_score": {"type": "number", "minimum": 0, "maximum": 10},
      "fairness_index": {"type": "number", "minimum": 0, "maximum": 10},
      "responsiveness_score": {"type": "number", "minimum": 0, "maximum": 10},
      "reliability_indicators": {
        "deadline_adherence": {"type": "number"},
        "fact_check_accuracy": {"type": "number"},
        "quote_verification": {"type": "number"},
        "embargo_compliance": {"type": "number"}
      },
      "communication_preferences": {
        "preferred_channels": [{"type": "string"}],
        "optimal_contact_times": [{"type": "string"}],
        "response_time_avg": {"type": "number"},
        "preferred_formats": [{"type": "string"}]
      }
    },
    "relationship_tracking": {
      "relationship_strength": {"type": "number", "minimum": 0, "maximum": 100},
      "interaction_history": [{
        "timestamp": {"type": "string", "format": "date-time"},
        "interaction_type": {"type": "string"},
        "outcome": {"type": "string"},
        "sentiment": {"type": "number"}
      }],
      "engagement_metrics": {
        "email_open_rate": {"type": "number"},
        "response_rate": {"type": "number"},
        "meeting_acceptance_rate": {"type": "number"}
      }
    },
    "metadata": {
      "created_at": {"type": "string", "format": "date-time"},
      "updated_at": {"type": "string", "format": "date-time"},
      "data_sources": [{"type": "string"}],
      "confidence_level": {"type": "number"},
      "privacy_compliance": {
        "gdpr_consent": {"type": "boolean"},
        "data_retention_date": {"type": "string", "format": "date"}
      }
    }
  },
  "required": ["journalist_id", "basic_info", "version"]
}