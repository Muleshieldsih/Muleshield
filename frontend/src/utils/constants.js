export const FRAUD_TYPES = ["UPI Fraud","Digital Arrest","Job Scam","Investment Scam","Romance Scam"]
export const BANKS = ["SBI","HDFC Bank","ICICI Bank","Axis Bank","PNB","Bank of Baroda","Canara Bank","Kotak Mahindra"]
export const CITIES = ["Delhi","Mumbai","Bengaluru","Hyderabad","Chennai","Kolkata","Pune","Jaipur","Lucknow","Noida","Gurgaon"]

export function amountFmt(n){ try{ return `₹${Number(n).toLocaleString('en-IN')}` }catch{ return `₹${n}` } }
export function pctFmt(p){ return `${(Number(p)*100).toFixed(1)}%` }

export const mockComplaints = [
  { ticket_id:'TKT-A1B2C3D4', victim_name:'Rohan Sharma', victim_bank:'HDFC Bank', victim_account:'XXXX-XXXX-4821', fraud_type:'UPI Fraud', stolen_amount:120000, city:'Delhi', state:'Delhi', complaint_timestamp: new Date(Date.now()-8*60000).toISOString(), status:'ACTIVE' },
  { ticket_id:'TKT-E5F6G7H8', victim_name:'Priya Nair', victim_bank:'SBI', victim_account:'XXXX-XXXX-9932', fraud_type:'Job Scam', stolen_amount:45000, city:'Mumbai', state:'Maharashtra', complaint_timestamp: new Date(Date.now()-22*60000).toISOString(), status:'ACTIVE' },
  { ticket_id:'TKT-I9J0K1L2', victim_name:'Amit Yadav', victim_bank:'ICICI Bank', victim_account:'XXXX-XXXX-7155', fraud_type:'Digital Arrest', stolen_amount:80000, city:'Pune', state:'Maharashtra', complaint_timestamp: new Date(Date.now()-41*60000).toISOString(), status:'ACTIVE' },
]

export const mockPrediction = {
  complaint_id:'TKT-A1B2C3D4',
  top3_atms:[
    { rank:1, atm_id:'ATM-SBI-004', confidence:0.924, lat:28.6139, lon:77.2090, bank:'SBI', address:'Connaught Place, Delhi', historical_fraud_count:12 },
    { rank:2, atm_id:'ATM-PNB-011', confidence:0.071, lat:28.6279, lon:77.2192, bank:'PNB', address:'Karol Bagh, Delhi', historical_fraud_count:7 },
    { rank:3, atm_id:'ATM-HDFC-089', confidence:0.018, lat:28.6051, lon:77.1892, bank:'HDFC Bank', address:'Chanakyapuri, Delhi', historical_fraud_count:3 },
  ],
  time_to_cashout_minutes: 27.4,
  interception_confidence: 0.924,
  inference_time_ms: 25.8,
  stolen_amount: 120000,
  terminal_account:'PNB-XXXX-0041',
  terminal_lat:28.6129, terminal_lon:77.2089
}
