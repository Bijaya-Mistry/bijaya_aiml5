  
import os
import uuid
import pandas as pd
from fastapi import UploadFile, HTTPException
import logging
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """
    Save an uploaded file to the specified destination.
    
    Args:
        upload_file: The uploaded file
        destination: The directory to save the file to
        
    Returns:
        The path to the saved file
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(destination, exist_ok=True)
        
        # Generate a unique filename to avoid collisions
        file_extension = os.path.splitext(upload_file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(destination, unique_filename)
        
        # Save the file
        content = await upload_file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        logger.info(f"File saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def get_file_path(filename: str, directory: str) -> str:
    """
    Get the full path to a file.
    
    Args:
        filename: The name of the file
        directory: The directory containing the file
        
    Returns:
        The full path to the file
    """
    return os.path.join(directory, filename)

def validate_csv_file(file_path: str) -> pd.DataFrame:
    """
    Validate a CSV file and return its contents as a DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    try:
        # Try to read the CSV file
        data = pd.read_csv(file_path)
        
        # Check if the file has any data
        if data.empty:
            raise HTTPException(status_code=400, detail="The CSV file is empty")
        
        logger.info(f"Successfully read CSV file with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty")
        raise HTTPException(status_code=400, detail="The CSV file is empty")
    
    except pd.errors.ParserError:
        logger.error("Error parsing the CSV file")
        raise HTTPException(status_code=400, detail="Error parsing the CSV file. Please check the format.")
    
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

def generate_sample_data(num_samples: int = 100) -> pd.DataFrame:
    """
    Generate sample banking data for testing purposes.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame containing sample banking data
    """
    import numpy as np
    import random
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Lists to store customer data
    data = []
    
    # Define possible values for categorical features
    education_levels = ['High School', 'Bachelors', 'Masters', 'PhD', 'Associate']
    occupation_categories = ['Professional', 'Engineer', 'Teacher', 'Doctor', 'Sales', 
                           'Administrative', 'Self-employed', 'Retired', 'Student', 'Other']
    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
    yes_no = ['Yes', 'No']
    account_types = ['Standard', 'Premium', 'Savings', 'Business']
    channels = ['Online', 'Mobile', 'Branch', 'ATM', 'Phone']
    risk_profiles = ['Conservative', 'Moderate', 'Aggressive']
    
    # Generate data for each customer
    for i in range(num_samples):
        # Basic demographics
        age = random.randint(18, 85)
        income = int(np.random.normal(70000, 30000))
        income = max(20000, income)
        
        # Select education and occupation
        education = random.choice(education_levels)
        occupation = random.choice(occupation_categories)
        marital_status = random.choice(marital_statuses)
        
        # Banking details
        years_as_customer = min(age - 18, random.randint(0, 30))
        account_balance = int(income * random.uniform(0.5, 3.0))
        credit_score = random.randint(500, 850)
        has_mortgage = random.choice(yes_no)
        has_loans = random.choice(yes_no)
        avg_monthly_transactions = random.randint(5, 100)
        account_type = random.choice(account_types)
        preferred_channel = random.choice(channels)
        savings_rate = round(random.uniform(0.01, 0.5), 2)
        investment_balance = int(account_balance * random.uniform(0, 3.0))
        loan_amount = 0 if has_loans == 'No' else int(income * random.uniform(0.5, 2.0))
        risk_profile = random.choice(risk_profiles)
        
        # Determine if customer is a potential investor
        # Higher probability for older, wealthier customers with higher savings rates
        investor_score = (
            (age / 85) * 0.2 + 
            (income / 200000) * 0.2 + 
            (account_balance / 300000) * 0.2 + 
            (savings_rate / 0.5) * 0.2 + 
            (investment_balance / 200000) * 0.2
        )
        is_investor = 1 if random.random() < investor_score else 0
        
        # Create customer record
        customer = {
            'age': age,
            'income': income,
            'education': education,
            'occupation': occupation,
            'marital_status': marital_status,
            'account_balance': account_balance,
            'credit_score': credit_score,
            'years_as_customer': years_as_customer,
            'has_mortgage': has_mortgage,
            'has_loans': has_loans,
            'avg_monthly_transactions': avg_monthly_transactions,
            'account_type': account_type,
            'preferred_channel': preferred_channel,
            'savings_rate': savings_rate,
            'investment_balance': investment_balance,
            'loan_amount': loan_amount,
            'risk_profile': risk_profile,
            'is_investor': is_investor
        }
        
        data.append(customer)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df