from constants import SYSTEM_PROMPT
import pandas as pd

def build_prompt_row_withoutDoc(row):
    field_descriptions = f"""
        The following is the structured input data for the claim:

        - Monthly Rent: {row['Monthly Rent']}
        - Amount of Claim: {row['Amount of Claim']}
        - Max Benefit: {row['Max Benefit']}
        - lease_duration_days: {row['lease_duration_days']}  # Total number of days in the lease agreement.
        - time_lived_before_moveout: {row['time_lived_before_moveout']}  # Number of days the tenant lived before moving out or eviction.
        - moved_out_early_flag: {row['moved_out_early_flag']}  # 1 if tenant moved out before lease end, 0 if not.
        - days_remaining_on_lease: {row['days_remaining_on_lease']}  # Days left on the lease when the claim was filed.
        - lease_month_start: {row['lease_month_start']}  # Month (1–12) the lease started.
        - lease_year_start: {row['lease_year_start']}  # Year the lease began.
        - num_tenants: {row['num_tenants']}  # Total number of tenants on the lease.
        - max_benefit_as_percent_of_rent: {row['max_benefit_as_percent_of_rent']}  # Maximum benefit as a percentage of monthly rent.
        - is_policy_generous: {row['is_policy_generous']}  # 1 if the policy is more generous than standard terms.
        - pm_claim_count: {row['pm_claim_count']}  # Number of past claims by this property manager.
        - is_repeat_pm: {row['is_repeat_pm']}  # 1 if this PM has filed previous claims; 0 otherwise.
        - termination_type_encoded: {row['termination_type_encoded']}  # Encoded reason for lease termination (e.g., early, eviction).
        - tenant_contact_encode: {row['tenant_contact_encode']}  # Encoded status of tenant contact (e.g., reached, unreachable).
        - second_tenant_encoded: {row['second_tenant_encoded']}  # 1 if a second tenant is listed; 0 otherwise.
        - third_tenant_encoded: {row['third_tenant_encoded']}  # 1 if a third tenant is listed; 0 otherwise.
        - lease_zip_freq: {row['lease_zip_freq']}
        - lease_state_freq: {row['lease_state_freq']}
        - lease_city_freq: {row['lease_city_freq']}

        WHAT IS THE PREDICTED APPROVED BENEFIT AMOUNT FOR THIS CLAIM? (STRICTLY Return only a single number Representing the Predicted APPROVED BENEFIT AMOUNT in dollars. Do not include any text, currency symbols, labels, formatting, or explanations.
        )
        """
    return field_descriptions.strip()

def build_prompt_row_withDoc(row):
    field_descriptions = f"""
        The following is the structured input data for the claim:

        - Monthly Rent: {row['Monthly Rent']}
        - Amount of Claim: {row['Amount of Claim']}
        - Max Benefit: {row['Max Benefit']}
        - lease_duration_days: {row['lease_duration_days']}  # Total number of days in the lease agreement.
        - time_lived_before_moveout: {row['time_lived_before_moveout']}  # Number of days the tenant lived before moving out or eviction.
        - moved_out_early_flag: {row['moved_out_early_flag']}  # 1 if tenant moved out before lease end, 0 if not.
        - days_remaining_on_lease: {row['days_remaining_on_lease']}  # Days left on the lease when the claim was filed.
        - lease_month_start: {row['lease_month_start']}  # Month (1–12) the lease started.
        - lease_year_start: {row['lease_year_start']}  # Year the lease began.
        - num_tenants: {row['num_tenants']}  # Total number of tenants on the lease.
        - max_benefit_as_percent_of_rent: {row['max_benefit_as_percent_of_rent']}  # Maximum benefit as a percentage of monthly rent.
        - is_policy_generous: {row['is_policy_generous']}  # 1 if the policy is more generous than standard terms.
        - pm_claim_count: {row['pm_claim_count']}  # Number of past claims by this property manager.
        - is_repeat_pm: {row['is_repeat_pm']}  # 1 if this PM has filed previous claims; 0 otherwise.
        - termination_type_encoded: {row['termination_type_encoded']}  # Encoded reason for lease termination (e.g., early, eviction).
        - tenant_contact_encode: {row['tenant_contact_encode']}  # Encoded status of tenant contact (e.g., reached, unreachable).
        - second_tenant_encoded: {row['second_tenant_encoded']}  # 1 if a second tenant is listed; 0 otherwise.
        - third_tenant_encoded: {row['third_tenant_encoded']}  # 1 if a third tenant is listed; 0 otherwise.
        - lease_zip_freq: {row['lease_zip_freq']}
        - lease_state_freq: {row['lease_state_freq']}
        - lease_city_freq: {row['lease_city_freq']}
        - Extracted Fields: {row['Extracted Fields']}

        WHAT IS THE PREDICTED APPROVED BENEFIT AMOUNT FOR THIS CLAIM? (STRICTLY Return only a single number Representing the Predicted APPROVED BENEFIT AMOUNT in dollars. Do not include any text, currency symbols, labels, formatting, or explanations.
        )
        """
    return field_descriptions.strip()

