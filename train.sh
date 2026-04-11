echo "==========================================="
echo " Starting Data Preprocessing Pipeline...   "
echo "==========================================="
python scripts/preprocess_data.py

echo "==========================================="
echo " Starting Model Training (Unsloth)...      "
echo "==========================================="
# Pass configuration path explicitly if needed, but our script currently hardcodes it, so just running python is fine
python scripts/train.py

echo "==========================================="
echo " Training Pipeline Finished Successfully!  "
echo "==========================================="