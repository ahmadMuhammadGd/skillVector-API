import dotenv
import os 
dotenv.load_dotenv()

fasttext_model      = os.getenv('COMPRESSED_FASTTEXT')
connection_string   = os.getenv('connection_string')
