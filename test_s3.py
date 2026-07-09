import os
import boto3
from dotenv import load_dotenv

# Force load the .env file explicitly from the current workspace directory
load_dotenv()

def quick_s3_test():
    print("☁️ Testing AWS S3 connection keys...")
    
    # Manually extract the keys from your saved .env file
    aws_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
    
    print(f"DEBUG: Found ID starting with: {str(aws_id)[:5]}...")
    print(f"DEBUG: Connecting to region: {aws_region}")
    
    try:
        # Pass the credentials explicitly into the client instance builder
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        
        response = s3.list_buckets()
        print("\n✅ SUCCESS! Your laptop successfully logged into AWS!")
        print("Here are your account buckets:")
        for bucket in response.get('Buckets', []):
            print(f" 🪣 -> {bucket['Name']}")
            
    except Exception as e:
        print(f"\n❌ FAILED: Authentication error. Details:\n{str(e)}")

if __name__ == "__main__":
    quick_s3_test()