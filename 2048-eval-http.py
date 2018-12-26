import tensorflow as tf
from BaseHTTPServer import BaseHTTPRequestHandler
import SocketServer
import json
import math
import time

PORT=8000 #to be non-privileged on linux

# Q-value
x=tf.placeholder(tf.float32,[None,4,4])
xp=tf.reshape(x,[-1,4,4,1])
cb=tf.Variable(tf.zeros([128]))
cw=tf.Variable(tf.truncated_normal([2,2,1,128],stddev=math.sqrt(2.0/(2*2*1+128))))
co=tf.nn.relu(tf.nn.conv2d(xp,cw,[1,1,1,1],'VALID')+cb)
cb2=tf.Variable(tf.zeros([64]))
cw2=tf.Variable(tf.truncated_normal([2,2,128,64],stddev=math.sqrt(2.0/(2*2*128+64))))
co2=tf.nn.relu(tf.nn.conv2d(co,cw2,[1,1,1,1],'VALID')+cb2)
xf=tf.reshape(co2,[-1,4*64])
b=tf.Variable(tf.zeros([128]))
w=tf.Variable(tf.truncated_normal([256,128],stddev=math.sqrt(2.0/(256+128))))
o=tf.nn.relu(tf.matmul(xf,w)+b)
#b1=tf.Variable(tf.zeros([32]))
#w1=tf.Variable(tf.truncated_normal([128,32],stddev=math.sqrt(2.0/(128+32))))
#o1=tf.nn.relu(tf.matmul(o,w1)+b1)
b2=tf.Variable(tf.zeros([4]))
w2=tf.Variable(tf.truncated_normal([128,4],stddev=math.sqrt(2.0/(128+4))))
y=tf.matmul(o,w2)+b2
yp=tf.placeholder(tf.float32,[None,4])
ymax=tf.reduce_max(y,1)
argmax=tf.argmax(y,1)

sess=tf.Session()
tf.train.Saver().restore(sess,'./model.1545678840.187.ckpt')

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            t=time.time()
            l=int(self.headers['Content-Length'])
            d=self.rfile.read(l)
            print(time.time()-t)
            t=time.time()
            dj=json.loads(d)
            tbl=[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
            for i in range(4):
                for j in range(4):
                    tmp=dj['rows'][i]['columns'][j]['val']
                    if tmp==0: tbl[i][j]=0.0
                    else: tbl[i][j]=math.log(tmp,2)
            print(tbl)
            #eval
            print(time.time()-t)
            t=time.time()
            action=sess.run([argmax],feed_dict={x:[tbl]})[0][0]
            print(action)
            print(time.time()-t)
            t=time.time()
            #action=1
            self.send_response(200)
            self.send_header('Content-type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'action':action}))
            self.wfile.write('\n')
            print(time.time()-t)
        except Exception, e:
            print(str(e))
            self.send_response(400)
            self.send_header('Content-type','text/plain')
            self.end_headers()
            self.wfile.write('wtf\n')

httpd=SocketServer.TCPServer(('',PORT),Handler)
try:
    httpd.serve_forever()
except:
    httpd.server_close()
