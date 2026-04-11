echo "==========================================="
echo " Running Inference Test Script...          "
echo "==========================================="
python scripts/inference.py

echo "==========================================="
echo " Starting Full Evaluation...               "
echo "==========================================="
python scripts/evaluate.py

echo "==========================================="
echo " Inference & Evaluation Complete!          "
echo "==========================================="