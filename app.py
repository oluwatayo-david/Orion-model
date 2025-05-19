import os
from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import PromptTemplate
import google.generativeai as genai
import redis
from cryptography.fernet import Fernet
import uuid
import json
import re  # Added this import for regex operations
from datetime import datetime, timedelta
from queue import Queue
import threading
import hashlib
import hmac
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ENCRYPTION_KEY = os.getenv("QUEUE_ENCRYPTION_KEY", Fernet.generate_key().decode())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_QUEUE_SIZE = 100  # Prevent overloading the system
PROCESSING_BATCH_SIZE = 5

# Initialize encryption
fernet = Fernet(ENCRYPTION_KEY.encode())

# Initialize Redis connection
redis_client = redis.Redis.from_url(REDIS_URL)

# Initialize processing queue and thread
processing_queue = Queue()
processing_lock = threading.Lock()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


# Initialize LLM components
def initialize_llm():
    # Create data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/placeholder.txt", "w") as f:
            f.write("Initial placeholder for document storage")

    documents = SimpleDirectoryReader("./data").load_data()

    Settings.llm = Gemini(
        api_key=GEMINI_API_KEY,
        model_name="models/gemini-1.5-flash",
        temperature=0.7,
    )

    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=GEMINI_API_KEY,
        title="financial_transactions"
    )

    index = VectorStoreIndex.from_documents(documents)

    template = (
        "You are a financial advisor analyzing transaction history. "
        "Address the user by their preferred title and name at the beginning of your response. "
        "Provide detailed analysis and recommendations based on the user's spending patterns. "
        "Focus only on financial advice. For non-financial questions, respond with: "
        "\"I can only provide financial advice based on your transaction history.\"\n"
        "Context:\n{context_str}\n"
        "Question: {query_str}\n"
        "Answer:"
    )

    qa_template = PromptTemplate(template)
    query_engine = index.as_query_engine(text_qa_template=qa_template)
    return query_engine


query_engine = initialize_llm()


# Secure Queue Implementation
class SecureTransactionQueue:
    def __init__(self):
        self.redis = redis_client
        self.queue_name = "financial_transactions"

    def _generate_hmac(self, data):
        """Generate HMAC for data integrity verification"""
        return hmac.new(
            ENCRYPTION_KEY.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def enqueue(self, user_id, transaction_data, user_name=None, user_title=None):
        """Securely add transaction to processing queue"""
        try:
            # Check queue size to prevent overload
            if self.redis.llen(self.queue_name) >= MAX_QUEUE_SIZE:
                return False, "Queue is full. Please try again later."

            # Create secure payload
            payload = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'user_name': user_name,
                'user_title': user_title,
                'timestamp': datetime.utcnow().isoformat(),
                'data': transaction_data
            }

            # Encrypt and sign
            encrypted = fernet.encrypt(json.dumps(payload).encode())
            signed_payload = {
                'payload': encrypted.decode(),
                'signature': self._generate_hmac(encrypted.decode())
            }

            # Add to queue
            self.redis.lpush(self.queue_name, json.dumps(signed_payload))
            return True, "Transaction queued successfully"

        except Exception as e:
            return False, f"Queue error: {str(e)}"

    def dequeue(self):
        """Securely retrieve transaction from queue"""
        try:
            item = self.redis.rpop(self.queue_name)
            if not item:
                return None

            signed_payload = json.loads(item)

            # Verify HMAC
            calculated_hmac = self._generate_hmac(signed_payload['payload'])
            if not hmac.compare_digest(calculated_hmac, signed_payload['signature']):
                raise ValueError("HMAC verification failed")

            # Decrypt
            decrypted = fernet.decrypt(signed_payload['payload'].encode())
            return json.loads(decrypted)

        except Exception as e:
            print(f"Dequeue error: {str(e)}")
            return None


# Store user profile information
def store_user_profile(user_id, name=None, title=None, gender=None):
    """Store user profile information in Redis"""
    try:
        key = f"user_profile:{user_id}"
        profile = {
            'user_id': user_id,
            'name': name,
            'title': title,
            'gender': gender,
            'updated_at': datetime.utcnow().isoformat()
        }
        redis_client.set(key, json.dumps(profile))
        return True
    except Exception as e:
        print(f"Error storing user profile: {str(e)}")
        return False


# Get user profile information
def get_user_profile(user_id):
    """Get user profile information from Redis"""
    try:
        key = f"user_profile:{user_id}"
        profile_data = redis_client.get(key)
        if profile_data:
            return json.loads(profile_data)
        return None
    except Exception as e:
        print(f"Error getting user profile: {str(e)}")
        return None


# Generate user greeting based on profile
def get_user_greeting(user_id):
    """Generate appropriate greeting for user"""
    profile = get_user_profile(user_id)

    if not profile or not profile.get('name'):
        return "Dear Customer"

    name = profile.get('name')
    title = profile.get('title')

    if title:
        return f"{title} {name}"

    # Determine appropriate title if not provided
    gender = profile.get('gender', '').lower()
    if gender == 'male':
        return f"Mr. {name}"
    elif gender == 'female':
        return f"Mrs. {name}"
    else:
        return f"Dear {name}"


# Background Processing Worker
def process_transactions_worker():
    """Background worker to process transactions"""
    queue = SecureTransactionQueue()
    while True:
        try:
            # Get batch of transactions
            batch = []
            for _ in range(PROCESSING_BATCH_SIZE):
                with processing_lock:
                    transaction = queue.dequeue()
                    if transaction:
                        batch.append(transaction)

            if not batch:
                time.sleep(5)  # Sleep if queue is empty
                continue

            # Process batch
            process_transaction_batch(batch)

        except Exception as e:
            print(f"Worker error: {str(e)}")
            time.sleep(10)


def process_transaction_batch(batch):
    """Process a batch of transactions"""
    try:
        # Group by user
        user_transactions = {}
        for transaction in batch:
            user_id = transaction['user_id']
            if user_id not in user_transactions:
                user_transactions[user_id] = []

            # Handle both single transaction or list of transactions
            if isinstance(transaction['data'], list):
                user_transactions[user_id].extend(transaction['data'])
            else:
                user_transactions[user_id].append(transaction['data'])

            # Store user profile information if available
            if 'user_name' in transaction and transaction['user_name']:
                store_user_profile(
                    user_id,
                    name=transaction.get('user_name'),
                    title=transaction.get('user_title')
                )

        # Print for debugging
        print(f"Processing batch for users: {list(user_transactions.keys())}")
        for user_id, txns in user_transactions.items():
            print(f"User {user_id} has {len(txns)} transactions")

        # Analyze each user's transactions
        for user_id, transactions in user_transactions.items():
            analyze_user_transactions(user_id, transactions)

    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        import traceback
        traceback.print_exc()


def analyze_user_transactions(user_id, transactions):
    """Analyze transactions and generate financial advice"""
    try:
        print(f"Analyzing transactions for user {user_id}: {transactions}")

        # Ensure transactions is properly formatted
        if isinstance(transactions, dict):
            # Convert dict to list if needed
            transactions = [transactions]

        # Convert transactions to analysis text - handle both string and dict formats for fields
        analysis_text = ""
        for t in transactions:
            # Extract date
            date = t.get('date') if isinstance(t, dict) else t[0] if isinstance(t, list) and len(
                t) > 0 else "Unknown date"

            # Extract amount
            amount = t.get('amount') if isinstance(t, dict) else t[1] if isinstance(t, list) and len(t) > 1 else "0"

            # Extract category
            category = t.get('category') if isinstance(t, dict) else t[2] if isinstance(t, list) and len(
                t) > 2 else "Unknown category"

            # Extract description
            description = t.get('description') if isinstance(t, dict) else t[3] if isinstance(t, list) and len(
                t) > 3 else "No description"

            # Add to analysis text
            analysis_text += f"{date} - {amount} - {category}: {description}\n"

        # Get time period for analysis
        time_period = get_analysis_time_period(transactions)

        # Get user greeting
        user_greeting = get_user_greeting(user_id)

        # Generate analysis
        prompt = (
            f"Address the user as '{user_greeting}' at the beginning of your response.\n"
            f"Analyze this user's transaction history from the past {time_period}:\n"
            f"{analysis_text}\n\n"
            "Provide:\n"
            "1. Spending pattern analysis\n"
            "2. Savings recommendations\n"
            "3. Potential budget optimizations\n"
            "4. Financial health assessment\n"
            "Format as a detailed financial report with clear sections."
        )

        analysis = query_engine.query(prompt)

        # Store analysis result
        store_analysis_result(user_id, {
            'timestamp': datetime.utcnow().isoformat(),
            'time_period': time_period,
            'transactions_count': len(transactions),
            'analysis': str(analysis),
            'user_greeting': user_greeting
        })

    except Exception as e:
        print(f"Analysis error for user {user_id}: {str(e)}")
        import traceback
        traceback.print_exc()


def get_analysis_time_period(transactions):
    """Determine appropriate time period for analysis"""
    if not transactions:
        return "1 month"  # Default

    try:
        dates = []
        for t in transactions:
            if isinstance(t, dict) and 'date' in t:
                date_str = t['date']
            elif isinstance(t, list) and len(t) > 0:
                date_str = t[0]
            else:
                continue

            # Handle different date formats
            try:
                # Try ISO format first
                date = datetime.fromisoformat(date_str)
            except ValueError:
                try:
                    # Try common formats
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                    except ValueError:
                        continue

            dates.append(date)

        if not dates:
            return "1 month"

        min_date = min(dates)
        max_date = max(dates)
        delta = max_date - min_date

        if delta.days <= 7:
            return "1 week"
        elif delta.days <= 30:
            return "1 month"
        elif delta.days <= 90:
            return "3 months"
        else:
            return f"{round(delta.days / 30)} months"

    except Exception as e:
        print(f"Error determining time period: {str(e)}")
        return "1 month"  # Default in case of error


def store_analysis_result(user_id, result):
    """Store analysis result in Redis"""
    try:
        key = f"financial_analysis:{user_id}"
        redis_client.lpush(key, json.dumps(result))
        redis_client.ltrim(key, 0, 9)  # Keep only last 10 analyses
    except Exception as e:
        print(f"Storage error: {str(e)}")


# Start background worker
worker_thread = threading.Thread(target=process_transactions_worker, daemon=True)
worker_thread.start()


# API Endpoints
@app.route('/api/user/profile', methods=['POST'])
def set_user_profile():
    """Set user profile information"""
    try:
        data = request.json
        user_id = data.get('user_id')
        name = data.get('name')
        title = data.get('title')  # Mr., Mrs., Dr., etc.
        gender = data.get('gender')  # male, female, other

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        success = store_user_profile(user_id, name, title, gender)

        if success:
            return jsonify({"status": "success", "message": "User profile updated"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to update profile"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/transactions', methods=['POST'])
def queue_transaction():
    """Endpoint to securely queue transactions"""
    try:
        data = request.json
        print(f"Received transaction data: {data}")

        # Handle 'query' format from Node.js app
        if 'query' in data:
            raw_text = data.get('query')

            # Extract user ID from the text
            user_id_match = re.search(r'User ID: ([a-f0-9]+)', raw_text)
            user_id = user_id_match.group(1) if user_id_match else 'unknown'

            # Extract name if available
            name_match = re.search(r'Name: ([A-Za-z\s]+)', raw_text)
            name = name_match.group(1).strip() if name_match else None

            # Extract title if available
            title_match = re.search(r'Title: (Mr\.|Mrs\.|Dr\.|Ms\.|Prof\.)', raw_text)
            title = title_match.group(1) if title_match else None

            # Parse notifications into transaction format
            pattern = r'Title: (.*)\nMessage: (.*)\nAction: (.*)\nDate: (.*)'
            matches = re.findall(pattern, raw_text)

            transactions = []
            for title_match, message, action, date in matches:
                # Extract amount from message if available
                amount_match = re.search(r'₦([0-9,]+)', message)
                amount = amount_match.group(1) if amount_match else '0'

                transactions.append({
                    'date': date,
                    'amount': amount,
                    'category': action,
                    'description': message
                })

        # Handle regular format with user_id and transactions
        else:
            user_id = data.get('user_id')
            transactions = data.get('transactions')
            name = data.get('user_name')
            title = data.get('user_title')

            if not user_id or not transactions:
                return jsonify({"error": "Missing user_id or transactions"}), 400

        print(f"Processed user_id: {user_id}, transactions: {transactions}")

        # If user profile data is provided, store it
        if name:
            store_user_profile(user_id, name, title)

        # Check format of transactions to ensure consistency
        if isinstance(transactions, list):
            # Validate transaction format
            for i, t in enumerate(transactions):
                if isinstance(t, dict):
                    required_fields = ['date', 'amount', 'category', 'description']
                    missing_fields = [field for field in required_fields if field not in t]
                    if missing_fields:
                        # Auto-fix missing fields with default values
                        for field in missing_fields:
                            transactions[i][field] = "Unknown" if field != 'amount' else "0"
                        print(f"Added default values for missing fields in transaction {i}")

        # Queue the transactions
        queue = SecureTransactionQueue()
        success, message = queue.enqueue(user_id, transactions, name, title)

        if success:
            return jsonify({"status": "success", "message": message}), 202
        else:
            return jsonify({"status": "error", "message": message}), 503

    except Exception as e:
        print(f"Transaction queuing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/analysis/latest', methods=['GET'])
def get_latest_analysis():
    """Get latest financial analysis for user"""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id parameter required"}), 400

        key = f"financial_analysis:{user_id}"
        result = redis_client.lrange(key, 0, 0)  # Get most recent

        if not result:
            return jsonify({"error": "No analysis available"}), 404

        return jsonify(json.loads(result[0])), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def financial_chat():
    """Financial advice chat endpoint"""
    try:
        data = request.json
        user_id = data.get('user_id')
        question = data.get('question')

        if not user_id or not question:
            return jsonify({"error": "Missing user_id or question"}), 400

        # Get user's greeting
        user_greeting = get_user_greeting(user_id)

        # Get user's recent transactions for context
        key = f"financial_analysis:{user_id}"
        transactions = redis_client.lrange(key, 0, 0)

        if not transactions:
            context = "No recent transaction history available."
        else:
            analysis = json.loads(transactions[0])
            context = f"Based on your financial activity over {analysis['time_period']}:\n{analysis['analysis']}"

        # Create personalized prompt
        prompt = (
            f"Address the user as '{user_greeting}' at the beginning of your response.\n"
            f"User question: {question}\n"
            f"Context from their transaction history:\n{context}\n"
            "Provide a detailed financial answer addressing their specific question. "
            "If the question is not financial-related, respond with: "
            "\"I can only provide financial advice based on your transaction history.\""
        )

        response = query_engine.query(prompt)
        return jsonify({"response": str(response)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        redis_client.ping()

        # Test Gemini connection
        test_llm = Gemini(api_key=GEMINI_API_KEY, model_name="gemini-1.5-flash")
        test_response = test_llm.complete("Hello")

        return jsonify({
            "status": "healthy",
            "redis": "connected",
            "gemini": "working",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500


if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")

    app.run(host='0.0.0.0', port=5000, threaded=True)


# import os
# from flask import Flask, request, jsonify
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.llms.gemini import Gemini
# from llama_index.embeddings.gemini import GeminiEmbedding
# from llama_index.core import PromptTemplate
# import google.generativeai as genai
# import redis
# from cryptography.fernet import Fernet
# import uuid
# import json
# import re  # Added this import for regex operations
# from datetime import datetime, timedelta
# from queue import Queue
# import threading
# import hashlib
# import hmac
# import time
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
#
# app = Flask(__name__)
#
# # Configuration
# REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# ENCRYPTION_KEY = os.getenv("QUEUE_ENCRYPTION_KEY", Fernet.generate_key().decode())
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# MAX_QUEUE_SIZE = 100  # Prevent overloading the system
# PROCESSING_BATCH_SIZE = 5
#
# # Initialize encryption
# fernet = Fernet(ENCRYPTION_KEY.encode())
#
# # Initialize Redis connection
# redis_client = redis.Redis.from_url(REDIS_URL)
#
# # Initialize processing queue and thread
# processing_queue = Queue()
# processing_lock = threading.Lock()
#
# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)
#
#
# # Initialize LLM components
# def initialize_llm():
#     # Create data directory if it doesn't exist
#     if not os.path.exists("./data"):
#         os.makedirs("./data")
#         with open("./data/placeholder.txt", "w") as f:
#             f.write("Initial placeholder for document storage")
#
#     documents = SimpleDirectoryReader("./data").load_data()
#
#     Settings.llm = Gemini(
#         api_key=GEMINI_API_KEY,
#         model_name="models/gemini-1.5-flash",
#         temperature=0.7,
#     )
#
#     Settings.embed_model = GeminiEmbedding(
#         model_name="models/embedding-001",
#         api_key=GEMINI_API_KEY,
#         title="financial_transactions"
#     )
#
#     index = VectorStoreIndex.from_documents(documents)
#
#     template = (
#         "You are a financial advisor analyzing transaction history. "
#         "Provide detailed analysis and recommendations based on the user's spending patterns. "
#         "Focus only on financial advice. For non-financial questions, respond with: "
#         "\"I can only provide financial advice based on your transaction history.\"\n"
#         "Context:\n{context_str}\n"
#         "Question: {query_str}\n"
#         "Answer:"
#     )
#
#     qa_template = PromptTemplate(template)
#     query_engine = index.as_query_engine(text_qa_template=qa_template)
#     return query_engine
#
#
# query_engine = initialize_llm()
#
#
# # Secure Queue Implementation
# class SecureTransactionQueue:
#     def __init__(self):
#         self.redis = redis_client
#         self.queue_name = "financial_transactions"
#
#     def _generate_hmac(self, data):
#         """Generate HMAC for data integrity verification"""
#         return hmac.new(
#             ENCRYPTION_KEY.encode(),
#             data.encode(),
#             hashlib.sha256
#         ).hexdigest()
#
#     def enqueue(self, user_id, transaction_data):
#         """Securely add transaction to processing queue"""
#         try:
#             # Check queue size to prevent overload
#             if self.redis.llen(self.queue_name) >= MAX_QUEUE_SIZE:
#                 return False, "Queue is full. Please try again later."
#
#             # Create secure payload
#             payload = {
#                 'id': str(uuid.uuid4()),
#                 'user_id': user_id,
#                 'timestamp': datetime.utcnow().isoformat(),
#                 'data': transaction_data
#             }
#
#             # Encrypt and sign
#             encrypted = fernet.encrypt(json.dumps(payload).encode())
#             signed_payload = {
#                 'payload': encrypted.decode(),
#                 'signature': self._generate_hmac(encrypted.decode())
#             }
#
#             # Add to queue
#             self.redis.lpush(self.queue_name, json.dumps(signed_payload))
#             return True, "Transaction queued successfully"
#
#         except Exception as e:
#             return False, f"Queue error: {str(e)}"
#
#     def dequeue(self):
#         """Securely retrieve transaction from queue"""
#         try:
#             item = self.redis.rpop(self.queue_name)
#             if not item:
#                 return None
#
#             signed_payload = json.loads(item)
#
#             # Verify HMAC
#             calculated_hmac = self._generate_hmac(signed_payload['payload'])
#             if not hmac.compare_digest(calculated_hmac, signed_payload['signature']):
#                 raise ValueError("HMAC verification failed")
#
#             # Decrypt
#             decrypted = fernet.decrypt(signed_payload['payload'].encode())
#             return json.loads(decrypted)
#
#         except Exception as e:
#             print(f"Dequeue error: {str(e)}")
#             return None
#
#
# # Background Processing Worker
# def process_transactions_worker():
#     """Background worker to process transactions"""
#     queue = SecureTransactionQueue()
#     while True:
#         try:
#             # Get batch of transactions
#             batch = []
#             for _ in range(PROCESSING_BATCH_SIZE):
#                 with processing_lock:
#                     transaction = queue.dequeue()
#                     if transaction:
#                         batch.append(transaction)
#
#             if not batch:
#                 time.sleep(5)  # Sleep if queue is empty
#                 continue
#
#             # Process batch
#             process_transaction_batch(batch)
#
#         except Exception as e:
#             print(f"Worker error: {str(e)}")
#             time.sleep(10)
#
#
# def process_transaction_batch(batch):
#     """Process a batch of transactions"""
#     try:
#         # Group by user
#         user_transactions = {}
#         for transaction in batch:
#             user_id = transaction['user_id']
#             if user_id not in user_transactions:
#                 user_transactions[user_id] = []
#
#             # Handle both single transaction or list of transactions
#             if isinstance(transaction['data'], list):
#                 user_transactions[user_id].extend(transaction['data'])
#             else:
#                 user_transactions[user_id].append(transaction['data'])
#
#         # Print for debugging
#         print(f"Processing batch for users: {list(user_transactions.keys())}")
#         for user_id, txns in user_transactions.items():
#             print(f"User {user_id} has {len(txns)} transactions")
#
#         # Analyze each user's transactions
#         for user_id, transactions in user_transactions.items():
#             analyze_user_transactions(user_id, transactions)
#
#     except Exception as e:
#         print(f"Batch processing error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#
# def analyze_user_transactions(user_id, transactions):
#     """Analyze transactions and generate financial advice"""
#     try:
#         print(f"Analyzing transactions for user {user_id}: {transactions}")
#
#         # Ensure transactions is properly formatted
#         if isinstance(transactions, dict):
#             # Convert dict to list if needed
#             transactions = [transactions]
#
#         # Convert transactions to analysis text - handle both string and dict formats for fields
#         analysis_text = ""
#         for t in transactions:
#             # Extract date
#             date = t.get('date') if isinstance(t, dict) else t[0] if isinstance(t, list) and len(
#                 t) > 0 else "Unknown date"
#
#             # Extract amount
#             amount = t.get('amount') if isinstance(t, dict) else t[1] if isinstance(t, list) and len(t) > 1 else "0"
#
#             # Extract category
#             category = t.get('category') if isinstance(t, dict) else t[2] if isinstance(t, list) and len(
#                 t) > 2 else "Unknown category"
#
#             # Extract description
#             description = t.get('description') if isinstance(t, dict) else t[3] if isinstance(t, list) and len(
#                 t) > 3 else "No description"
#
#             # Add to analysis text
#             analysis_text += f"{date} - {amount} - {category}: {description}\n"
#
#         # Get time period for analysis
#         time_period = get_analysis_time_period(transactions)
#
#         # Generate analysis
#         prompt = (
#             f"Analyze this user's transaction history from the past {time_period}:\n"
#             f"{analysis_text}\n\n"
#             "Provide:\n"
#             "1. Spending pattern analysis\n"
#             "2. Savings recommendations\n"
#             "3. Potential budget optimizations\n"
#             "4. Financial health assessment\n"
#             "Format as a detailed financial report with clear sections."
#         )
#
#         analysis = query_engine.query(prompt)
#
#         # Store analysis result
#         store_analysis_result(user_id, {
#             'timestamp': datetime.utcnow().isoformat(),
#             'time_period': time_period,
#             'transactions_count': len(transactions),
#             'analysis': str(analysis)
#         })
#
#     except Exception as e:
#         print(f"Analysis error for user {user_id}: {str(e)}")
#         import traceback
#         traceback.print_exc()
#
#
# def get_analysis_time_period(transactions):
#     """Determine appropriate time period for analysis"""
#     if not transactions:
#         return "1 month"  # Default
#
#     try:
#         dates = []
#         for t in transactions:
#             if isinstance(t, dict) and 'date' in t:
#                 date_str = t['date']
#             elif isinstance(t, list) and len(t) > 0:
#                 date_str = t[0]
#             else:
#                 continue
#
#             # Handle different date formats
#             try:
#                 # Try ISO format first
#                 date = datetime.fromisoformat(date_str)
#             except ValueError:
#                 try:
#                     # Try common formats
#                     date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
#                 except ValueError:
#                     try:
#                         date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
#                     except ValueError:
#                         continue
#
#             dates.append(date)
#
#         if not dates:
#             return "1 month"
#
#         min_date = min(dates)
#         max_date = max(dates)
#         delta = max_date - min_date
#
#         if delta.days <= 7:
#             return "1 week"
#         elif delta.days <= 30:
#             return "1 month"
#         elif delta.days <= 90:
#             return "3 months"
#         else:
#             return f"{round(delta.days / 30)} months"
#
#     except Exception as e:
#         print(f"Error determining time period: {str(e)}")
#         return "1 month"  # Default in case of error
#
#
#
# def store_analysis_result(user_id, result):
#     """Store analysis result in Redis"""
#     try:
#         key = f"financial_analysis:{user_id}"
#         redis_client.lpush(key, json.dumps(result))
#         redis_client.ltrim(key, 0, 9)  # Keep only last 10 analyses
#     except Exception as e:
#         print(f"Storage error: {str(e)}")
#
#
# # Start background worker
# worker_thread = threading.Thread(target=process_transactions_worker, daemon=True)
# worker_thread.start()
#
#
# # API Endpoints
# @app.route('/api/transactions', methods=['POST'])
# def queue_transaction():
#     """Endpoint to securely queue transactions"""
#     try:
#         data = request.json
#         print(f"Received transaction data: {data}")
#
#         # Handle 'query' format from Node.js app
#         if 'query' in data:
#             raw_text = data.get('query')
#
#             # Extract user ID from the text
#             user_id_match = re.search(r'User ID: ([a-f0-9]+)', raw_text)
#             user_id = user_id_match.group(1) if user_id_match else 'unknown'
#
#             # Parse notifications into transaction format
#             pattern = r'Title: (.*)\nMessage: (.*)\nAction: (.*)\nDate: (.*)'
#             matches = re.findall(pattern, raw_text)
#
#             transactions = []
#             for title, message, action, date in matches:
#                 # Extract amount from message if available
#                 amount_match = re.search(r'₦([0-9,]+)', message)
#                 amount = amount_match.group(1) if amount_match else '0'
#
#                 transactions.append({
#                     'date': date,
#                     'amount': amount,
#                     'category': action,
#                     'description': message
#                 })
#
#         # Handle regular format with user_id and transactions
#         else:
#             user_id = data.get('user_id')
#             transactions = data.get('transactions')
#
#             if not user_id or not transactions:
#                 return jsonify({"error": "Missing user_id or transactions"}), 400
#
#         print(f"Processed user_id: {user_id}, transactions: {transactions}")
#
#         # Check format of transactions to ensure consistency
#         if isinstance(transactions, list):
#             # Validate transaction format
#             for i, t in enumerate(transactions):
#                 if isinstance(t, dict):
#                     required_fields = ['date', 'amount', 'category', 'description']
#                     missing_fields = [field for field in required_fields if field not in t]
#                     if missing_fields:
#                         # Auto-fix missing fields with default values
#                         for field in missing_fields:
#                             transactions[i][field] = "Unknown" if field != 'amount' else "0"
#                         print(f"Added default values for missing fields in transaction {i}")
#
#         # Queue the transactions
#         queue = SecureTransactionQueue()
#         success, message = queue.enqueue(user_id, transactions)
#
#         if success:
#             return jsonify({"status": "success", "message": message}), 202
#         else:
#             return jsonify({"status": "error", "message": message}), 503
#
#     except Exception as e:
#         print(f"Transaction queuing error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/api/analysis/latest', methods=['GET'])
# def get_latest_analysis():
#     """Get latest financial analysis for user"""
#     try:
#         user_id = request.args.get('user_id')
#         if not user_id:
#             return jsonify({"error": "user_id parameter required"}), 400
#
#         key = f"financial_analysis:{user_id}"
#         result = redis_client.lrange(key, 0, 0)  # Get most recent
#
#         if not result:
#             return jsonify({"error": "No analysis available"}), 404
#
#         return jsonify(json.loads(result[0])), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/chat', methods=['POST'])
# def financial_chat():
#     """Financial advice chat endpoint"""
#     try:
#         data = request.json
#         user_id = data.get('user_id')
#         question = data.get('question')
#
#         if not user_id or not question:
#             return jsonify({"error": "Missing user_id or question"}), 400
#
#         # Get user's recent transactions for context
#         key = f"financial_analysis:{user_id}"
#         transactions = redis_client.lrange(key, 0, 0)
#
#         if not transactions:
#             context = "No recent transaction history available."
#         else:
#             analysis = json.loads(transactions[0])
#             context = f"Based on your financial activity over {analysis['time_period']}:\n{analysis['analysis']}"
#
#         # Create personalized prompt
#         prompt = (
#             f"User question: {question}\n"
#             f"Context from their transaction history:\n{context}\n"
#             "Provide a detailed financial answer addressing their specific question. "
#             "If the question is not financial-related, respond with: "
#             "\"I can only provide financial advice based on your transaction history.\""
#         )
#
#         response = query_engine.query(prompt)
#         return jsonify({"response": str(response)}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     try:
#         # Test Redis connection
#         redis_client.ping()
#
#         # Test Gemini connection
#         test_llm = Gemini(api_key=GEMINI_API_KEY, model_name="gemini-1.5-flash")
#         test_response = test_llm.complete("Hello")
#
#         return jsonify({
#             "status": "healthy",
#             "redis": "connected",
#             "gemini": "working",
#             "timestamp": datetime.utcnow().isoformat()
#         }), 200
#     except Exception as e:
#         return jsonify({
#             "status": "unhealthy",
#             "error": str(e),
#             "timestamp": datetime.utcnow().isoformat()
#         }), 500
#
#
# if __name__ == '__main__':
#     # Create data directory if it doesn't exist
#     if not os.path.exists("./data"):
#         os.makedirs("./data")
#
#     app.run(host='0.0.0.0', port=5000, threaded=True)