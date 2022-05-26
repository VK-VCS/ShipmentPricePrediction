import pickle
import pandas as pd
import xgboost

class PricingModel:
    cols = ['Unit of Measure (Per Pack)', 'Pack Price', 'Unit Price',
       'Weight (Kilograms)', 'Line Item Insurance (USD)',
       'Managed By_Ethiopia Field Office', 'Managed By_Haiti Field Office',
       'Managed By_PMO - US', 'Managed By_South Africa Field Office',
       'Fulfill Via_Direct Drop', 'Fulfill Via_From RDC',
       'Vendor INCO Term_CIF', 'Vendor INCO Term_CIP', 'Vendor INCO Term_DAP',
       'Vendor INCO Term_DDP', 'Vendor INCO Term_DDU', 'Vendor INCO Term_EXW',
       'Vendor INCO Term_FCA', 'Vendor INCO Term_N/A - From RDC',
       'Shipment Mode_Air', 'Shipment Mode_Air Charter', 'Shipment Mode_Ocean',
       'Shipment Mode_Truck', 'Product Group_ACT', 'Product Group_ANTM',
       'Product Group_ARV', 'Product Group_HRDT', 'Product Group_MRDT',
       'Sub Classification_ACT', 'Sub Classification_Adult',
       'Sub Classification_HIV test',
       'Sub Classification_HIV test - Ancillary', 'Sub Classification_Malaria',
       'Sub Classification_Pediatric', 'First Line Designation_No',
       'First Line Designation_Yes']

    def __init__(self, d):
        self.params = d

    def check_mandatory_fields(self):
        msg = ''
        for c in self.params.keys():
            for i in self.cols:
                if i not in self.params[c].keys():
                    msg = f'Fields missing in row {c}, expected columns :{len(self.cols)}, columns received :{len(self.params[c].keys())}'
                    break
        return msg

    def model_selection(self):
        model = pickle.load(open('Models/XGBoost.sav', 'rb'))
        return model

    def predict(self):
        model = self.model_selection()
        data = pd.read_json(self.params).T
        print(data)
        predictions = model.predict(data)
        return pd.to_json(predictions)
        #predictions = model.predict([self.params.values()])




