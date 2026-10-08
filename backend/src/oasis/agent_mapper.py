"""
OASIS Agent Mapper - Maps FHIR R4 JKN entities (Patient, Doctor, Hospital) to OASIS Agent Profiles.
"""
from typing import Dict, List, Any, Optional


class JKNToOASISMapper:
    """Maps JKN entities into OASIS social/autonomous agent profiles"""

    def __init__(self):
        self.agent_registry = {}

    def map_patient_to_agent(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a JKN Patient to an OASIS User Agent
        """
        p_id = str(patient_data.get("pasien_id", patient_data.get("patient_id", "P01")))
        p_name = patient_data.get("pasien_raw", f"Pasien {p_id}")
        complaint = patient_data.get("alasan_berobat", "Pemeriksaan rutin")
        feedback = patient_data.get("umpan_balik_pasien", "Pelayanan baik")
        sentiment = float(patient_data.get("skor_sentimen", 0.0))
        fraud_risk = float(patient_data.get("skor_fraud", 0.0))

        profile = {
            "agent_id": f"agent_patient_{p_id}",
            "entity_type": "patient",
            "name": p_name,
            "fhir_resource": "Patient",
            "medical_complaint": complaint,
            "patient_feedback": feedback,
            "sentiment_score": sentiment,
            "active_status": "in_treatment" if fraud_risk < 0.7 else "home_unaware",
            "behavior": {
                "interaction_probability": 0.3,
                "reporting_honesty": 0.95,
                "fraud_propensity": fraud_risk
            }
        }
        self.agent_registry[profile["agent_id"]] = profile
        return profile

    def map_doctor_to_agent(self, doctor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a JKN Doctor to an OASIS Practitioner Agent
        """
        d_id = str(doctor_data.get("dokter_id", doctor_data.get("doctor_id", "D01")))
        d_name = doctor_data.get("dokter_raw", f"Dokter {d_id}")
        faskes_id = doctor_data.get("faskes_id", "RS_A")
        specialty = doctor_data.get("specialty", "Spesialis Bedah")
        fraud_risk = float(doctor_data.get("skor_fraud", 0.0))

        profile = {
            "agent_id": f"agent_doctor_{d_id}",
            "entity_type": "doctor",
            "name": d_name,
            "fhir_resource": "Practitioner",
            "specialty": specialty,
            "hospital_affiliation": faskes_id,
            "behavior": {
                "clinical_volume_per_day": 25,
                "upcoding_opportunism": 0.85 if fraud_risk > 0.6 else 0.05,
                "phantom_billing_tendency": 0.80 if fraud_risk > 0.8 else 0.02,
                "fraud_propensity": fraud_risk
            }
        }
        self.agent_registry[profile["agent_id"]] = profile
        return profile

    def map_hospital_to_agent(self, hospital_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a JKN Hospital to an OASIS Organization/Moderator Agent
        """
        h_id = str(hospital_data.get("faskes_id", hospital_data.get("hospital_id", "RS_A")))
        fraud_risk = float(hospital_data.get("skor_fraud", 0.0))

        profile = {
            "agent_id": f"agent_hospital_{h_id}",
            "entity_type": "hospital",
            "name": f"Rumah Sakit {h_id}",
            "fhir_resource": "Organization",
            "bed_capacity": 150,
            "behavior": {
                "claim_batch_frequency_hours": 12,
                "verification_strictness": 0.35 if fraud_risk > 0.6 else 0.92,
                "vclaim_integration_active": True,
                "fraud_propensity": fraud_risk
            }
        }
        self.agent_registry[profile["agent_id"]] = profile
        return profile
