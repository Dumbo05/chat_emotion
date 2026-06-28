import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
root=Path('models/speech');split_path=root/'wavlm_clean_split_v2.json';split=json.loads(split_path.read_text(encoding='utf8'));assert split['test_status']=='sealed-not-evaluated'
w=json.loads((root/'wavlm_clean_v2_frozen_configs.json').read_text(encoding='utf8'))['frozen_wavlm_configs'];m=json.loads((root/'mfcc_clean_v2_frozen_config.json').read_text(encoding='utf8'))['frozen_mfcc_config']
models=[{'id':'mfcc_rbf_svm','display':'MFCC + RBF-SVM','feature':'MFCC handcrafted full 177','classifier':'RBF-SVM balanced','config':m},{'id':'wavlm_mean','display':'WavLM mean pooling','feature':'WavLM mean','classifier':'RBF-SVM','config':w['mean_unbalanced']},{'id':'wavlm_mean_std','display':'WavLM mean+std pooling','feature':'WavLM mean+std','classifier':'RBF-SVM','config':w['mean_std_unbalanced']},{'id':'wavlm_mean_std_balanced','display':'WavLM mean+std + balanced SVM','feature':'WavLM mean+std','classifier':'RBF-SVM balanced','config':w['mean_std_balanced']}]
plan={'protocol':split['protocol'],'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'split_manifest_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'selection_data':'train and validation only','test_status_before_run':split['test_status'],'final_refit_policy':'refit each frozen configuration on train + validation speakers, then evaluate all four pre-registered comparators in one test opening','models':models,'rejected_pooling_by_validation':w['std_balanced']}
out=root/'wavlm_clean_v2_frozen_evaluation_plan.json'
if out.exists():raise SystemExit('冻结计划已存在，拒绝覆盖')
out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'plan':str(out),'models':[x['id'] for x in models],'test_status':split['test_status']},ensure_ascii=False,indent=2))
