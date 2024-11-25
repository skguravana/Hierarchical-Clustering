from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from cluster_utils import calculate_cluster

with open('calculate_cluster', 'rb') as f:
    calculate_cluster = pickle.load(f)


# Load pre-trained scaler and PCA
with open('aug_scaler', 'rb') as f:
    scaler = pickle.load(f)

with open('aug_pca', 'rb') as f:
    pca = pickle.load(f)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
   
        # Extract numeric inputs
        year = float(request.form.get('year', 0))
        selling_price = float(request.form.get('selling_price', 0))
        km_driven = float(request.form.get('km_driven', 0))
        to_be_scaled_atts = [year, selling_price, km_driven]

        other_atts = []

        fuels = ['Diesel', 'Electric', 'LPG', 'Petrol']
        for fuel_type in fuels:
            other_atts.append(1 if request.form.get('fuel') == fuel_type else 0)

        sellers = ['Individual', 'Trustmark Dealer']
        for seller_type in sellers:
            other_atts.append(1 if request.form.get('seller_type') == seller_type else 0)

        other_atts.append(1 if request.form.get('transmission') == 'Manual' else 0)

        owners = ['Fourth & Above Owner', 'Second Owner', 'Test Drive Car', 'Third Owner']
        for owner_type in owners:
            other_atts.append(1 if request.form.get('owner') == owner_type else 0)

        
        scaled_atts = scaler.transform([to_be_scaled_atts])  # Scale numerical attributes
        combined_atts = scaled_atts[0].tolist()
        combined_atts+=other_atts

        new_point = np.array(combined_atts, dtype=float)
        new_point = new_point.reshape(1, -1)
        new_point_pca = pca.transform(new_point)

        combined_atts+=new_point_pca[0].tolist()

        cluster = str(calculate_cluster(combined_atts))
        
        

   
        return render_template('index.html',cluster=cluster)

    

if __name__ == '__main__':
    app.run(debug=True)
