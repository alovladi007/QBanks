import os, json, psycopg2, uuid
from rq import get_current_job
from analytics.calibration.sh_core import load_pool, iterative_sh, upsert_k

def calibrate_job(exam_code, dsn, tau, n, test_len, iters, alpha, theta_dist, floor, ceil, topic_tau, topic_weights, dry_run, run_id=None):
    job = get_current_job()
    job.meta.update({"state":"running","current_iter":0,"total_iters":iters}); job.save_meta()
    conn=psycopg2.connect(dsn); cur=conn.cursor()
    if run_id is None:
        run_id=str(uuid.uuid4())
        cur.execute("INSERT INTO calibration_runs(id,exam_code,status,params,created_at,started_at) VALUES (%s,%s,%s,%s,now(),now())",
          (run_id, exam_code, "running", json.dumps({"tau":tau,"n":n,"test_len":test_len,"iters":iters,"alpha":alpha,"theta_dist":theta_dist,"floor":floor,"ceil":ceil,"topic_tau":topic_tau,"topic_weights":topic_weights,"dry_run":dry_run})))
        conn.commit()
    else:
        cur.execute("UPDATE calibration_runs SET status='running', started_at=now() WHERE id=%s",(run_id,)); conn.commit()
    try:
        pool=load_pool(conn, exam_code)
        if not pool:
            job.meta.update({"state":"empty"}); job.save_meta()
            cur.execute("UPDATE calibration_runs SET status='empty', finished_at=now(), result=%s WHERE id=%s", (json.dumps({"updated":0,"history":[],"diff":[]}), run_id)); conn.commit()
            cur.close(); conn.close(); return {"updated":0,"history":[],"diff":[]}
        before=[{"qid":it["qid"],"ver":it["ver"],"sh_p":float(it["k"])} for it in pool]
        kmap={ (it["qid"],it["ver"]):float(it["k"]) for it in pool }
        history=[]
        for t in range(iters):
            km,seen,hist=iterative_sh(pool,tau,n,test_len,1,alpha,theta_dist,floor,ceil,None,topic_tau,topic_weights)
            for it in pool: it["k"]=km[(it["qid"],it["ver"])]
            kmap=km; history.extend(hist)
            job.meta.update({"current_iter":t+1,"avg_exp":hist[-1]["avg_exp"],"max_over":hist[-1]["max_over"]}); job.save_meta()
            cur.execute("UPDATE calibration_runs SET history = COALESCE(history,'[]'::jsonb) || %s::jsonb WHERE id=%s", (json.dumps([hist[-1]]), run_id)); conn.commit()
        after=[{"qid":it["qid"],"ver":it["ver"],"sh_p":float(kmap[(it["qid"],it["ver"])])} for it in pool]
        amap={(a["qid"],a["ver"]):a["sh_p"] for a in after}
        diff=[{"qid":b["qid"],"ver":b["ver"],"before":float(b["sh_p"]), "after": float(amap.get((b["qid"],b["ver"]), b["sh_p"])), "delta": float(amap.get((b["qid"],b["ver"]), b["sh_p"]))-float(b["sh_p"])} for b in before]
        if not dry_run: upsert_k(conn,kmap)
        result={"updated":len(kmap),"history":history,"diff":diff}
        cur.execute("UPDATE calibration_runs SET status='done', finished_at=now(), result=%s WHERE id=%s",(json.dumps(result),run_id)); conn.commit()
        job.meta.update({"state":"done"}); job.save_meta()
        cur.close(); conn.close(); return result
    except Exception as e:
        job.meta.update({"state":"failed"}); job.save_meta()
        cur.execute("UPDATE calibration_runs SET status='failed', finished_at=now(), error=%s WHERE id=%s",(str(e),run_id)); conn.commit()
        cur.close(); conn.close(); raise