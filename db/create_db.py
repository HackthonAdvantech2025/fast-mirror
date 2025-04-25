import sqlite3
import json
import os

def create_db():
    print("Creating database...")
    # 連接到 SQLite 資料庫（如果不存在，會自動建立一個新的)
    db_path = os.path.expanduser('~/fast_mirror/db/data.db')
    conn = sqlite3.connect(db_path)
    # 創建一個遊標物件，用來執行 SQL 指令
    cursor = conn.cursor()

    # 檢查資料庫中是不是已經建立 product_info 表
    res = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='product_info'")
    if res.fetchone() is None:
        # 創建一個資料表來儲存 JSON 檔案的資料
        cursor.execute('''CREATE TABLE product_info (
                        product_id TEXT PRIMARY KEY,
                        name TEXT,
                        src TEXT
                        
                    )''')
        print("Table 'product_info' created.")
    else:
        print("Table 'product_info' already exists.")

    # 讀取 JSON 檔案並插入資料到 SQLite 資料庫
    json_path = os.path.expanduser('~/fast_mirror/db/product_info.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'product_id' in item and 'name' in item and 'src' in item:
                        try:
                            cursor.execute(
                                'INSERT OR IGNORE INTO product_info (product_id, name, src) VALUES (?, ?, ?)',
                                (item['product_id'], item['name'], item['src'])
                            )
                        except sqlite3.IntegrityError:
                            print(f"Product ID {item['product_id']} already exists. Skipping.")
                        except Exception as e:
                            print(f"Error inserting item {item.get('product_id', 'N/A')}: {e}")
                    else:
                        print(f"Skipping invalid item: {item}")
            else:
                print("JSON data is not a list of products.")

        # 提交變更
        conn.commit()
        print("Data inserted successfully (or skipped existing).")

    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 關閉連接
        conn.close()
        print("Database connection closed.")