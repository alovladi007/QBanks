import math, random, statistics
import psycopg2, psycopg2.extras
D = 1.7
def logistic(x): return 1/(1+math.exp(-x))
def prob_3pl(theta,a,b,c): return c + (1-c)*logistic(D*a*(theta-b))
def fisher_info_3pl(theta,a,b,c):
    P=prob_3pl(theta,a,b,c); Q=1-P
    if P<=0 or Q<=0 or (1-c)<=0: return 0.0
    return (D**2)*(a**2)*(Q/P)*((P-c)/(1-c))**2
def sample_theta(dist="normal0,1"):
    return random.gauss(0,1) if dist.startswith("normal") else (random.uniform(-1,1) if dist.startswith("uniform") else 0.0)
def load_pool(conn, exam_code):
    sql = '''
    SELECT qv.question_id, qv.version, qv.topic_id,
           COALESCE(ic.a,1.0) a, COALESCE(ic.b,0.0) b, COALESCE(ic.c,0.2) c,
           COALESCE(iec.sh_p,1.0) sh_p
    FROM question_publications qp
    JOIN question_versions qv ON qv.question_id=qp.question_id AND qv.version=qp.live_version
    LEFT JOIN item_calibration ic ON ic.question_id=qv.question_id AND ic.version=qv.version AND ic.model='3pl'
    LEFT JOIN item_exposure_control iec ON iec.question_id=qv.question_id AND iec.version=qv.version
    WHERE qp.exam_code=%s AND qv.state='published'
    '''
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql,(exam_code,)); rows=cur.fetchall()
    return [{"qid":int(r["question_id"]),"ver":int(r["version"]),"topic":r["topic_id"],"a":float(r["a"]),"b":float(r["b"]),"c":float(r["c"]),"k":float(r["sh_p"]) if r["sh_p"] is not None else 1.0} for r in rows]
def simulate_once(pool,n,test_len,theta_dist,use_k=True,seed=None):
    if seed is not None: random.seed(seed)
    seen={ (it["qid"],it["ver"]):0 for it in pool }
    for _ in range(n):
        theta=sample_theta(theta_dist); administered=set()
        for _pos in range(test_len):
            cand=[it for it in pool if (it["qid"],it["ver"]) not in administered]
            if not cand: break
            scored=sorted(((fisher_info_3pl(theta,it["a"],it["b"],it["c"]),it) for it in cand), key=lambda x:x[0], reverse=True)
            chosen=None
            for _,it in scored:
                if (not use_k) or (random.random()<=max(0.0,min(1.0,it["k"]))):
                    chosen=it; break
            if chosen is None: chosen=scored[0][1]
            administered.add((chosen["qid"],chosen["ver"]))
        for key in administered: seen[key]+=1
    return seen, {}
def _compute_topic_tau(tau, topic_tau, topic_weights):
    if topic_tau: return {str(k): float(v) for k,v in topic_tau.items()}
    if topic_weights:
        s=sum(float(v) for v in topic_weights.values()) or 1.0
        return {str(k): tau*(float(v)/s) for k,v in topic_weights.items()}
    return None
def iterative_sh(pool,tau,n,test_len,iters,alpha,theta_dist,floor,ceil,seed=None,topic_tau=None,topic_weights=None):
    tmap=_compute_topic_tau(tau,topic_tau,topic_weights)
    k={ (it["qid"],it["ver"]):float(it["k"]) for it in pool }
    history=[]
    for t in range(iters):
        for it in pool: it["k"]=k[(it["qid"],it["ver"])]
        seen,_=simulate_once(pool,n,test_len,theta_dist,use_k=True,seed=None if seed is None else seed+t)
        r={ key: seen[key]/max(1,n) for key in seen }
        newk={}
        for it in pool:
            key=(it["qid"],it["ver"]); ri=r.get(key,0.0); ki=k[key]
            cap = float(tmap.get(str(it["topic"]),tau)) if tmap is not None else tau
            if ri<=0.0: val=min(1.0,max(floor,ki*1.1))
            else:
                ratio=cap/ri; val=ki*(ratio**alpha); val=min(1.0,max(floor,min(ceil,val)))
            newk[key]=val
        k=newk
        avg_exp=(sum(r.values())/len(r)) if r else 0.0
        if tmap is None:
            max_over=max((ri-tau for ri in r.values()), default=0.0)
        else:
            ovs=[]
            for it in pool:
                key=(it["qid"],it["ver"]); cap=float(tmap.get(str(it["topic"]),tau)); ovs.append(r.get(key,0.0)-cap)
            max_over=max(ovs) if ovs else 0.0
        history.append({"iter":t+1,"avg_exp":avg_exp,"max_over":max_over})
    return k, seen, history
def upsert_k(conn,kmap):
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public")
        for (qid,ver),kval in kmap.items():
            cur.execute(
              '''INSERT INTO item_exposure_control(question_id,version,sh_p)
                 VALUES (%s,%s,%s)
                 ON CONFLICT (question_id,version) DO UPDATE SET sh_p=EXCLUDED.sh_p, updated_at=now()''',
              (qid,ver,float(kval))
            )
    conn.commit()