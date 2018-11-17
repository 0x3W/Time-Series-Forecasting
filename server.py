import time
import socket
import json
import ForecastingLib as mli
fo = open('data.log', 'w')
fo2 = open('data-' + str(time.time()) + '.log', 'w')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 8887))
s.listen(10)

def processRequest(jreq):
    req = json.loads(jreq)
    timestamp = req['timestamp']
    target = req['target']['name']
    features = [ f['name'] for f in req['features'] ]
    print("timestamp: " + str(timestamp) + ", target: " + target + ", features: [" + ' '.join(features) + "]")
    resp = { 'timestamp': timestamp, 'target': target, 'uniPred': [0.0], 'multiPred':[0.0] }
    return json.dumps(resp)

def handleConnection(s,mod,modScale,sgdScale,ForMod):
    global fo
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    f = conn.makefile()
    requests = []
    l = f.readline().rstrip()
    while l != '':
        requests.append(l)
        fo.write(l + "\n")
        fo.flush()
        l = f.readline().rstrip()
    for req in requests:
        resp = processRequest(req)
        mod, resp2 = mli.process(smth, mod,modScale,sgdScale,ForMod)
        print(resp2)
        fo2.write(json.dumps(resp2) + '\n')
        fo2.flush()
        conn.sendall((json.dumps(resp2) + '\n').encode())
    conn.close()
    return mod

mod = 0
sgdScale=0
ForMod=0
modScale = 0
while 1:
    print('starting..')
    mod = handleConnection(s,mod,modScale,sgdScale,ForMod)
    fo.write("\n")
    fo.flush()
s.close()
fo.close()
fo2.close()
