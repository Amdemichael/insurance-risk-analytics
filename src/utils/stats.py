def load_data(file_path):
    """Load data from a pipe-separated text file and handle basic data types.
    
    Args:
        file_path (str): Path to the input text file.
        
    Returns:
        pd.DataFrame: Loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(file_path, sep='|', encoding='utf-8')
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    for col in ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm', 
                'CustomValueEstimate', 'CapitalOutstanding', 'kilowatts', 'cubiccapacity', 
                'RegistrationYear']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df