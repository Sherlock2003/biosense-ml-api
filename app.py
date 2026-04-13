# ============================================================
#   BioSense ML API v2 - Disease Prediction Backend
#   Deploy on Render.com
# ============================================================
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np, requests, os
from datetime import datetime
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

TS_CH  = os.environ.get("THINGSPEAK_CHANNEL_ID",  "3317281")
TS_KEY = os.environ.get("THINGSPEAK_READ_API_KEY", "GE7EN76H12BQVULN")

LABELS   = {0:"Normal",1:"Mild Risk",2:"Fever Suspected",3:"Low SpO2 - Hypoxia",4:"Cardiac Anomaly",5:"Fall Detected",6:"Multiple Risk Factors"}
SEVERITY = {0:"normal",1:"warning",2:"warning",3:"danger",4:"danger",5:"danger",6:"danger"}
history  = deque(maxlen=50)

def gen_data(n=6000):
    np.random.seed(42); data=[]
    choices=['normal','fever','hypoxia','tachy','brady','fall','multi']
    probs=[.50,.15,.10,.08,.07,.05,.05]
    for _ in range(n):
        c=np.random.choice(choices,p=probs)
        if   c=='normal':  hr,s,t=np.random.normal(75,8),np.random.normal(98,1),np.random.normal(36.5,.3);  ax,ay,az=np.random.normal(0,.5),np.random.normal(0,.5),np.random.normal(9.8,.3);  st,f,lb=np.random.randint(0,300),0,0
        elif c=='fever':   hr,s,t=np.random.normal(90,10),np.random.normal(97,1),np.random.normal(38.5,.5); ax,ay,az=np.random.normal(0,.4),np.random.normal(0,.4),np.random.normal(9.8,.3);  st,f,lb=np.random.randint(0,100),0,2
        elif c=='hypoxia': hr,s,t=np.random.normal(88,12),np.random.normal(91,2),np.random.normal(36.6,.4); ax,ay,az=np.random.normal(0,.6),np.random.normal(0,.6),np.random.normal(9.8,.5);  st,f,lb=np.random.randint(0,50),0,3
        elif c=='tachy':   hr,s,t=np.random.normal(115,10),np.random.normal(97,1.5),np.random.normal(36.8,.4);ax,ay,az=np.random.normal(0,1),np.random.normal(0,1),np.random.normal(9.8,.5); st,f,lb=np.random.randint(50,500),0,4
        elif c=='brady':   hr,s,t=np.random.normal(42,5),np.random.normal(96,2),np.random.normal(36.3,.4);  ax,ay,az=np.random.normal(0,.3),np.random.normal(0,.3),np.random.normal(9.8,.2); st,f,lb=np.random.randint(0,30),0,4
        elif c=='fall':    hr,s,t=np.random.normal(95,15),np.random.normal(96,2),np.random.normal(36.5,.5); ax,ay,az=np.random.normal(0,8),np.random.normal(0,8),np.random.normal(9.8,8);    st,f,lb=np.random.randint(0,50),1,5
        else:              hr,s,t=np.random.normal(108,12),np.random.normal(92,2),np.random.normal(38,.8);  ax,ay,az=np.random.normal(0,2),np.random.normal(0,2),np.random.normal(9.8,2);    st,f,lb=np.random.randint(0,100),np.random.choice([0,1]),6
        am=np.sqrt(ax**2+ay**2+az**2)
        data.append([max(0,hr),max(0,min(100,s)),max(30,t),ax,ay,az,max(0,st),f,am,1 if 60<=hr<=100 else 0,1 if s<94 else 0,lb])
    return data

def train_model():
    print("[BioSense] Training ML model...")
    raw=gen_data()
    X=np.array([r[:11] for r in raw]); y=np.array([r[11] for r in raw])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    m=RandomForestClassifier(n_estimators=200,max_depth=12,class_weight='balanced',random_state=42,n_jobs=-1)
    m.fit(Xtr,ytr)
    acc=round(m.score(Xte,yte)*100,1)
    print(f"[BioSense] Model ready. Accuracy: {acc}%")
    return m, sc, acc

MODEL, SCALER, ACC = train_model()

def run_predict(hr,spo2,temp,ax,ay,az,steps,fall):
    am=np.sqrt(ax**2+ay**2+az**2)
    feats=np.array([[hr,spo2,temp,ax,ay,az,steps,fall,am,1 if 60<=hr<=100 else 0,1 if spo2<94 else 0]])
    fs=SCALER.transform(feats)
    label=int(MODEL.predict(fs)[0])
    proba=MODEL.predict_proba(fs)[0]
    classes=list(MODEL.classes_)
    ci=classes.index(label) if label in classes else 0
    conf=round(float(proba[ci])*100,1)
    full={LABELS[i]:round(float(proba[classes.index(i)])*100,1) if i in classes else 0.0 for i in range(7)}
    alerts=[]
    if temp>37.5: alerts.append("High temperature — possible fever")
    if spo2>0 and spo2<94: alerts.append("Low SpO2 — possible hypoxia")
    if hr>100: alerts.append("Tachycardia detected")
    if hr>0 and hr<50: alerts.append("Bradycardia detected")
    if fall==1: alerts.append("Fall event detected")
    result={
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "inputs":    {"heart_rate":round(hr,1),"spo2":round(spo2,1),"skin_temp":round(temp,1),"accel_x":round(ax,3),"accel_y":round(ay,3),"accel_z":round(az,3),"steps":int(steps),"fall_detected":int(fall)},
        "prediction":{"label":label,"condition":LABELS[label],"severity":SEVERITY[label],"confidence":conf,"probabilities":full},
        "alerts":    alerts
    }
    history.appendleft(result)
    return result

@app.route('/')
def home():
    return jsonify({"service":"BioSense ML API","version":"2.0","status":"running","model_accuracy":f"{ACC}%","college":"GPREC ECE Batch 21, Kurnool","channel":"ThingSpeak 3317281","endpoints":{"/predict":"GET","/predict/manual":"POST","/history":"GET","/health":"GET"}})

@app.route('/health')
def health():
    return jsonify({"status":"ok","timestamp":datetime.utcnow().isoformat(),"accuracy":f"{ACC}%"})

@app.route('/predict')
def predict_live():
    try:
        url=f"https://api.thingspeak.com/channels/{TS_CH}/feeds/last.json?api_key={TS_KEY}"
        d=requests.get(url,timeout=10).json()
        return jsonify(run_predict(float(d.get('field1')or 0),float(d.get('field2')or 0),float(d.get('field3')or 0),float(d.get('field4')or 0),float(d.get('field5')or 0),float(d.get('field6')or 0),int(float(d.get('field7')or 0)),int(float(d.get('field8')or 0))))
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route('/predict/manual',methods=['POST'])
def predict_manual():
    try:
        d=request.get_json()
        return jsonify(run_predict(float(d.get('heart_rate',75)),float(d.get('spo2',98)),float(d.get('skin_temp',36.5)),float(d.get('accel_x',0)),float(d.get('accel_y',0)),float(d.get('accel_z',9.8)),int(d.get('steps',0)),int(d.get('fall_detected',0))))
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route('/history')
def get_history():
    return jsonify({"count":len(history),"history":list(history)})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=False)
