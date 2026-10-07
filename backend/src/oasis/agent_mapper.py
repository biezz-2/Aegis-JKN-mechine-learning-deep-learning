"""
OASIS Agent Mapper - Map JKN entities to OASIS agent profiles
"""
from typing import Dict, List, Any
import json


class JKNToOASISMapper:
    """Map JKN entities (patients, doctors, hospitals) to OASIS agents"""

    def __init__(self):
        self.agent_profiles = []

    def map_patient_to_agent(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a JKN patient to an OASIS user agent

        Args:
            patient_data: Patient data from JKN dataset

        Returns:
            OASIS agent profile
        """
        agent_profile = {
            "agent_id": patient_data.get("patient_id", f"user_{len(self.agent_profiles)}"),
            "agent_type": "user",
            "name": patient_data.get("name", f"Patient {patient_data.get('patient_id', 'Unknown')}"),
            "age": patient_data.get("age", 45),
            "gender": patient_data.get("gender", "unspecified"),
            "location": patient_data.get("location", "Indonesia"),
            "bio": f"JKN patient with {patient_data.get('claims_count', 0)} claims",
            "interests": ["healthcare", "insurance"],
            "personality": {
                "openness": 0.6,
                "conscientiousness": 0.7,
                "extraversion": 0.5,
                "agreeableness": 0.6,
                "neuroticism": 0.4
            },
            "behavior": {
                "posting_frequency": 0.1,
                "interaction_probability": 0.2,
                "fraud_propensity": patient_data.get("fraud_risk", 0.0)
            }
        }
        return agent_profile

    def map_doctor_to_agent(self, doctor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a JKN doctor to an OASIS influencer agent

        Args:
            doctor_data: Doctor data from JKN dataset

        Returns:
            OASIS agent profile
        """
        agent_profile = {
            "agent_id": doctor_data.get("doctor_id", f"doctor_{len(self.agent_profiles)}"),
            "agent_type": "influencer",
            "name": doctor_data.get("name", f"Dr. {doctor_data.get('doctor_id', 'Unknown')}"),
            "specialty": doctor_data.get("specialty", "General Practitioner"),
            "experience_years": doctor_data.get("experience", 10),
            "hospital": doctor_data.get("hospital", "Unknown"),
            "bio": f"Medical doctor specializing in {doctor_data.get('specialty', 'general practice')}",
            "followers": doctor_data.get("patient_count", 100),
            "influence_score": doctor_data.get("influence", 0.7),
            "behavior": {
                "posting_frequency": 0.3,
                "interaction_probability": 0.5,
                "fraud_propensity": doctor_data.get("fraud_risk", 0.0)
            }
        }
        return agent_profile

    def map_hospital_to_agent(self, hospital_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a JKN hospital to an OASIS platform moderator agent

        Args:
            hospital_data: Hospital data from JKN dataset

        Returns:
            OASIS agent profile
        """
        agent_profile = {
            "agent_id": hospital_data.get("hospital_id", f"hospital_{len(self.agent_profiles)}"),
            "agent_type": "moderator",
            "name": hospital_data.get("name", f"Hospital {hospital_data.get('hospital_id', 'Unknown')}"),
            "type": hospital_data.get("type", "General Hospital"),
            "location": hospital_data.get("location", "Indonesia"),
            "capacity": hospital_data.get("beds", 100),
            "bio": f"{hospital_data.get('type', 'Hospital')} with {hospital_data.get('beds', 100)} beds",
            "behavior": {
                "moderation_activity": 0.8,
                "approval_rate": hospital_data.get("approval_rate", 0.9),
                "fraud_propensity": hospital_data.get("fraud_risk", 0.0)
            }
        }
        return agent_profile

    def generate_agent_profiles(
        self,
        patients: List[Dict[str, Any]],
        doctors: List[Dict[str, Any]],
        hospitals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate OASIS agent profiles from JKN entities

        Args:
            patients: List of patient data
            doctors: List of doctor data
            hospitals: List of hospital data

        Returns:
            List of OASIS agent profiles
        """
        self.agent_profiles = []

        # Map patients
        for patient in patients:
            self.agent_profiles.append(self.map_patient_to_agent(patient))

        # Map doctors
        for doctor in doctors:
            self.agent_profiles.append(self.map_doctor_to_agent(doctor))

        # Map hospitals
        for hospital in hospitals:
            self.agent_profiles.append(self.map_hospital_to_agent(hospital))

        return self.agent_profiles

    def save_profiles(self, filepath: str):
        """Save agent profiles to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.agent_profiles, f, indent=2)

    def load_profiles(self, filepath: str) -> List[Dict[str, Any]]:
        """Load agent profiles from JSON file"""
        with open(filepath, 'r') as f:
            self.agent_profiles = json.load(f)
        return self.agent_profiles


if __name__ == "__main__":
    # Test mapper
    mapper = JKNToOASISMapper()

    # Sample data
    patients = [{"patient_id": "P001", "name": "John Doe", "age": 45, "fraud_risk": 0.1}]
    doctors = [{"doctor_id": "D001", "name": "Dr. Smith", "specialty": "Cardiology", "fraud_risk": 0.05}]
    hospitals = [{"hospital_id": "H001", "name": "City Hospital", "type": "General", "fraud_risk": 0.02}]

    profiles = mapper.generate_agent_profiles(patients, doctors, hospitals)
    print(f"Generated {len(profiles)} agent profiles")
    print(json.dumps(profiles[0], indent=2))
