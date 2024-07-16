import requests

def check_open_access(doi):
    """
    Check if the DOI is available for open access using Unpaywall.
    """
    url = f"https://api.unpaywall.org/v2/{doi}?email=lionnist127@gmail.com"  # Replace 'YOUR_EMAIL' with your actual email address
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        data = response.json()  # Convert response to JSON
        
        if data['is_oa']:
            print(f"Open Access available: {data['best_oa_location']['url_for_pdf']}")
        else:
            print("This article is not available for open access.")
    except requests.RequestException as e:
        print(f"Failed to retrieve data: {e}")

# Example DOI
doi = '10.1111/j.1432-1033.1977.tb11235.x'  # Example DOI, replace with the DOI you need to check
check_open_access(doi)
