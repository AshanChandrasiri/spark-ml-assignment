pip install -r ./requirements.txt

python main.py

# Create Virtual environment
python -m venv venv

# Activate the virtual environment
source venv/Scripts/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r ./requirements.txt

echo "Running main.py..."
python ./main.py