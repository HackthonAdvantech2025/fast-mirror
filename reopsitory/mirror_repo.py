import sqlite3
import os

DB_PATH = os.path.expanduser('~/fast_mirror/db/data.db')

# 將連接移至函數內部，避免模組載入時就連接
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        # 可以設定 row_factory 使回傳結果為字典形式，更方便使用
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as e:
        print(f"Error connecting to database at {DB_PATH}: {e}")
        # 可以在這裡引發自訂例外或回傳 None，視應用程式需求而定
        raise  # 重新引發例外，讓上層知道連接失敗

def get_product_info():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT product_id, name, src FROM product_info")
        products = cursor.fetchall()
        # 將 sqlite3.Row 物件轉換為字典列表
        return [dict(row) for row in products]
    except Exception as e:
        print(f"Error fetching product info: {e}")
        return {"error": "Could not fetch product information"}
    finally:
        if conn:
            conn.close()
