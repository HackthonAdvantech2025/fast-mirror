import os
import boto3

async def get_product_info():
    dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
    table = dynamodb.Table('GenAI_2025')
    try:
        response = table.scan()
        items = response.get('Items', [])
        return items
    except Exception as e:
        print(f"Error fetching product info: {e}")
        return {"error": "Could not fetch product information"}