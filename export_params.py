import joblib, json

# load your pipeline
pipe = joblib.load('bot_click_model.pkl')

# extract encodings/scaling
cat_enc = pipe.named_steps['pre'].named_transformers_['cat']
num_scl = pipe.named_steps['pre'].named_transformers_['num']
logreg  = pipe.named_steps['clf']

params = {
  # list of categories for each categorical feature
  "cat_features": {
    "ip_address":      cat_enc.categories_[0].tolist(),
    "device":          cat_enc.categories_[1].tolist(),
    "browser":         cat_enc.categories_[2].tolist(),
    "geo_location":    cat_enc.categories_[3].tolist(),
    "referrer":        cat_enc.categories_[4].tolist()
  },
  # scaler means & scales in the same order as your numeric_features list
  "num_mean":  num_scl.mean_.tolist(),
  "num_scale": num_scl.scale_.tolist(),
  # logistic regression weights and intercept
  "coeff":     logreg.coef_[0].tolist(),
  "intercept": float(logreg.intercept_[0])
}

with open('model_params.json','w') as f:
    json.dump(params, f)

print("Wrote model_params.json")
