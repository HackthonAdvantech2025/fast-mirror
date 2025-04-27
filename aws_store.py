import boto3
import uuid

# 初始化 DynamoDB 與 S3 客戶端
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
s3 = boto3.client('s3', region_name='ap-northeast-1')

DYNAMODB_TABLE = 'Products'
S3_BUCKET = 'your-s3-bucket-name'

def upload_image_to_s3(file_path, object_name=None):
    if object_name is None:
        object_name = f'products/{uuid.uuid4()}_{file_path.split("/")[-1]}'
    s3.upload_file(file_path, S3_BUCKET, object_name)
    url = f'https://{S3_BUCKET}.s3.amazonaws.com/{object_name}'
    return url

def save_product_to_dynamodb(product_id, name, description, image_url):
    table = dynamodb.Table(DYNAMODB_TABLE)
    table.put_item(
        Item={
            'product_id': product_id,
            'name': name,
            'description': description,
            'image_url': image_url
        }
    )

# 範例：上傳圖片並儲存商品資料
def add_product(file_path, name, description):
    product_id = str(uuid.uuid4())
    image_url = upload_image_to_s3(file_path)
    save_product_to_dynamodb(product_id, name, description, image_url)
    return product_id, image_url
