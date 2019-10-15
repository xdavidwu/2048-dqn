#include "2048.h"
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <experimental/algorithm>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/framework/gradients.h>
#include <absl/strings/string_view.h>

#include <sys/time.h>

using namespace tensorflow;
using namespace tensorflow::ops;

#define MEM_SIZE	1<<17
#define MEM_START_TRAIN	1<<17
#define BATCH	1<<9
#define QT_UPDATE_INT	64
#define PRINT_INT	(1L<<16)
#define DOUBLE_DQN	0
#define GAMMA	0.94f
#define LEARNING_RATE	0.000005
//static float LEARNING_RATE=1;
typedef struct{
	cell_t s[16];
	dir_t a;
	score_t r;
	cell_t sp[16];
	int isterm;
}mem_t;

static std::vector<mem_t> replaymem(MEM_SIZE);
static int replaymem_idx=0,replaymem_full=0;

static float epsilon;
static int train_counter=0;
static int print_counter=0;
static int block_train=0;

static Scope scope=Scope::NewRootScope();
static Placeholder x(scope,DT_FLOAT);
static Placeholder qt_x(scope,DT_FLOAT);
static Placeholder yp(scope,DT_FLOAT);

static Variable cb(scope,{256},DT_FLOAT);
static Variable cw(scope,{2,2,1,256},DT_FLOAT);
static Variable cb2(scope,{64},DT_FLOAT);
static Variable cw2(scope,{2,2,256,64},DT_FLOAT);
static Variable b(scope,{128},DT_FLOAT);
static Variable w(scope,{256,128},DT_FLOAT);
static Variable b2(scope,{4},DT_FLOAT);
static Variable w2(scope,{128,4},DT_FLOAT);

static Variable cb_ms(scope,{256},DT_FLOAT);
static Variable cw_ms(scope,{2,2,1,256},DT_FLOAT);
static Variable cb2_ms(scope,{64},DT_FLOAT);
static Variable cw2_ms(scope,{2,2,256,64},DT_FLOAT);
static Variable b_ms(scope,{128},DT_FLOAT);
static Variable w_ms(scope,{256,128},DT_FLOAT);
static Variable b2_ms(scope,{4},DT_FLOAT);
static Variable w2_ms(scope,{128,4},DT_FLOAT);
static Variable cb_mom(scope,{256},DT_FLOAT);
static Variable cw_mom(scope,{2,2,1,256},DT_FLOAT);
static Variable cb2_mom(scope,{64},DT_FLOAT);
static Variable cw2_mom(scope,{2,2,256,64},DT_FLOAT);
static Variable b_mom(scope,{128},DT_FLOAT);
static Variable w_mom(scope,{256,128},DT_FLOAT);
static Variable b2_mom(scope,{4},DT_FLOAT);
static Variable w2_mom(scope,{128,4},DT_FLOAT);

static Variable qt_cb(scope,{256},DT_FLOAT);
static Variable qt_cw(scope,{2,2,1,256},DT_FLOAT);
static Variable qt_cb2(scope,{64},DT_FLOAT);
static Variable qt_cw2(scope,{2,2,256,64},DT_FLOAT);
static Variable qt_b(scope,{128},DT_FLOAT);
static Variable qt_w(scope,{256,128},DT_FLOAT);
static Variable qt_b2(scope,{4},DT_FLOAT);
static Variable qt_w2(scope,{128,4},DT_FLOAT);

static Reshape xp(scope,x,{-1,4,4,1});
static Relu co(scope,Add(scope,Conv2D(scope,xp,cw,{1,1,1,1},absl::string_view("VALID")),cb));
static Relu co2(scope,Add(scope,Conv2D(scope,co,cw2,{1,1,1,1},absl::string_view("VALID")),cb2));
static Reshape xf(scope,co2,{-1,4*64});
static Relu o(scope,Add(scope,MatMul(scope,xf,w),b));
static Add y(scope,MatMul(scope,o,w2),b2);
static Max ymax(scope,y,1);
static ArgMax argmax(scope,y,1);
static Mean loss(scope,Mean(scope,Square(scope,Subtract(scope,yp,y)),1),0);

static Reshape qt_xp(scope,qt_x,{-1,4,4,1});
static Relu qt_co(scope,Add(scope,Conv2D(scope,qt_xp,qt_cw,{1,1,1,1},absl::string_view("VALID")),qt_cb));
static Relu qt_co2(scope,Add(scope,Conv2D(scope,qt_co,qt_cw2,{1,1,1,1},absl::string_view("VALID")),qt_cb2));
static Reshape qt_xf(scope,qt_co2,{-1,4*64});
static Relu qt_o(scope,Add(scope,MatMul(scope,qt_xf,qt_w),qt_b));
static Add qt_y(scope,MatMul(scope,qt_o,qt_w2),qt_b2);
static Max qt_ymax(scope,qt_y,1);
static std::vector<Output> qt_update({
		Assign(scope,qt_cb,cb),
		Assign(scope,qt_cw,cw),
		Assign(scope,qt_cb2,cb2),
		Assign(scope,qt_cw2,cw2),
		Assign(scope,qt_b,b),
		Assign(scope,qt_w,w),
		Assign(scope,qt_b2,b2),
		Assign(scope,qt_w2,w2)
		});

static std::vector<Output> q_init({
		Assign(scope,cb,Fill(scope,{256},float(0.0))),
		Assign(scope,cw,Multiply(scope,TruncatedNormal(scope,{2,2,1,256},DT_FLOAT),Fill(scope,{2,2,1,256},float(sqrt(2.0/(2*2*1+256)))))),
		Assign(scope,cb2,Fill(scope,{64},float(0.0))),
		Assign(scope,cw2,Multiply(scope,TruncatedNormal(scope,{2,2,256,64},DT_FLOAT),Fill(scope,{2,2,256,64},float{sqrt(2.0/(2*2*256+64))}))),
		Assign(scope,b,Fill(scope,{128},float(0.0))),
		Assign(scope,w,Multiply(scope,TruncatedNormal(scope,{256,128},DT_FLOAT),Fill(scope,{256,128},float(sqrt(2.0/(256+128)))))),
		Assign(scope,b2,Fill(scope,{4},float(0.0))),
		Assign(scope,w2,Multiply(scope,TruncatedNormal(scope,{128,4},DT_FLOAT),Fill(scope,{128,4},float(sqrt(2.0/(128+4))))))
		});
static std::vector<Output> rms_init({
		Assign(scope,cb_ms,ZerosLike(scope,cb)),
		Assign(scope,cw_ms,ZerosLike(scope,cw)),
		Assign(scope,cb2_ms,ZerosLike(scope,cb2)),
		Assign(scope,cw2_ms,ZerosLike(scope,cw2)),
		Assign(scope,b_ms,ZerosLike(scope,b)),
		Assign(scope,w_ms,ZerosLike(scope,w)),
		Assign(scope,b2_ms,ZerosLike(scope,b2)),
		Assign(scope,w2_ms,ZerosLike(scope,w2)),
		Assign(scope,cb_mom,ZerosLike(scope,cb)),
		Assign(scope,cw_mom,ZerosLike(scope,cw)),
		Assign(scope,cb2_mom,ZerosLike(scope,cb2)),
		Assign(scope,cw2_mom,ZerosLike(scope,cw2)),
		Assign(scope,b_mom,ZerosLike(scope,b)),
		Assign(scope,w_mom,ZerosLike(scope,w)),
		Assign(scope,b2_mom,ZerosLike(scope,b2)),
		Assign(scope,w2_mom,ZerosLike(scope,w2))
		});

static std::vector<Output> step;

static ClientSession* cs;

// 2048 max 1<<17 max r < 1<<18 /4 1<<16
static float log2tbl[1<<16];

static std::thread* prefetch_thread=nullptr;
static std::vector<mem_t> bat(BATCH);
static std::vector<mem_t>::iterator it;

static float loss_sum=0.0;

//static long us_sum=0;

static Tensor *s,*sp;
static Tensor *s1=new Tensor(DT_FLOAT, TensorShape{BATCH,4,4}),*sp1=new Tensor(DT_FLOAT, TensorShape{BATCH,4,4});
static Tensor *s2=new Tensor(DT_FLOAT, TensorShape{BATCH,4,4}),*sp2=new Tensor(DT_FLOAT, TensorShape{BATCH,4,4});

static inline void gen_log2(){
	for(int a=4;a<(1<<18);a+=4) log2tbl[a>>2]=std::log2f(a);
}

static inline float dbl_rand(){
	return (float)rand()/RAND_MAX;
}

void dqnagent_pushmem(cell_t *s,dir_t a,score_t r,cell_t *sp,int isterm){
	if(unlikely(block_train))return;
	mem_t nmem={.a=a,.r=r,.isterm=isterm};
	memcpy(nmem.s,s,sizeof(cell_t)*16);
	memcpy(nmem.sp,sp,sizeof(cell_t)*16);
	replaymem[replaymem_idx++]=nmem;
	if(unlikely(replaymem_idx==MEM_SIZE)){
		replaymem_full=1;
		replaymem_idx=0;
	}
}

static void dqnagent_init(){
	SessionOptions opt;
	//opt.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.9);
	//opt.config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
	cs=new ClientSession(scope,opt);
	TF_CHECK_OK(cs->Run({},q_init,nullptr));
	TF_CHECK_OK(cs->Run({},rms_init,nullptr));
	TF_CHECK_OK(cs->Run({},{qt_update},nullptr));
	std::vector<Output> grad_outputs;
	TF_CHECK_OK(AddSymbolicGradients(scope,{loss},{cb,cw,cb2,cw2,b,w,b2,w2},&grad_outputs));
	Input::Initializer lr(float(LEARNING_RATE)),rho(float(0.9)),ep(float(1.0e-10)),mom(float(0.0));	
	step.push_back(ApplyRMSProp(scope,cb,cb_ms,cb_mom,lr,rho,mom,ep,{grad_outputs[0]}));
	step.push_back(ApplyRMSProp(scope,cw,cw_ms,cw_mom,lr,rho,mom,ep,{grad_outputs[1]}));
	step.push_back(ApplyRMSProp(scope,cb2,cb2_ms,cb2_mom,lr,rho,mom,ep,{grad_outputs[2]}));
	step.push_back(ApplyRMSProp(scope,cw2,cw2_ms,cw2_mom,lr,rho,mom,ep,{grad_outputs[3]}));
	step.push_back(ApplyRMSProp(scope,b,b_ms,b_mom,lr,rho,mom,ep,{grad_outputs[4]}));
	step.push_back(ApplyRMSProp(scope,w,w_ms,w_mom,lr,rho,mom,ep,{grad_outputs[5]}));
	step.push_back(ApplyRMSProp(scope,b2,b2_ms,b2_mom,lr,rho,mom,ep,{grad_outputs[6]}));
	step.push_back(ApplyRMSProp(scope,w2,w2_ms,w2_mom,lr,rho,mom,ep,{grad_outputs[7]}));
}

Tensor x_d(DT_FLOAT, TensorShape{1,4,4});
auto x_df=x_d.flat<float>().data();

dir_t dqnagent_getact(cell_t *s){
	if(unlikely(dbl_rand()<epsilon)){
		return static_cast<dir_t>(rand()&0x3);
	}
	for(int a=0;a<16;a++) x_df[a]=(float)s[a];
	std::vector<Tensor> outputs;
	TF_CHECK_OK(cs->Run({{x,x_d}},{argmax},&outputs));
	return static_cast<dir_t>(outputs[0].scalar<int>()());
}

static void sample_thr(){
	if(unlikely(replaymem_full)) it=replaymem.end();
	else it=replaymem.begin()+replaymem_idx-1;
	std::experimental::sample(replaymem.begin(),it,bat.begin(),BATCH,std::mt19937_64{std::random_device{}()});
	auto mf=((print_counter&0x1)?s1:s2)->flat<float>().data();
	auto mfp=((print_counter&0x1)?sp1:sp2)->flat<float>().data();
#pragma omp parallel for num_threads(2)
	for(int i=0;i<BATCH;i++){
		for(int j=0;j<16;j++){
			mf[i*16+j]=(float)bat[i].s[j];
			mfp[i*16+j]=(float)bat[i].sp[j];
		}
	}
}

static Tensor q_,ya,ynxt,ym;
//struct timeval tp,ta;

void dqnagent_train(){
	if(unlikely(block_train||((!replaymem_full)&&replaymem_idx<MEM_START_TRAIN))) return;
	if(unlikely(prefetch_thread==nullptr)){
		if(replaymem_full) it=replaymem.end();
		else it=replaymem.begin()+replaymem_idx-1;
		std::experimental::sample(replaymem.begin(),it,bat.begin(),BATCH,std::mt19937_64{std::random_device{}()});
		auto mf=((print_counter&0x1)?s1:s2)->flat<float>().data();
		auto mfp=((print_counter&0x1)?sp1:sp2)->flat<float>().data();
#pragma omp parallel for num_threads(2)
		for(int i=0;i<BATCH;i++){
			for(int j=0;j<16;j++){
				mf[i*16+j]=(float)bat[i].s[j];
				mfp[i*16+j]=(float)bat[i].sp[j];
			}
		}
	}
	else{
		//gettimeofday(&tp,NULL);
		prefetch_thread->join();
		delete prefetch_thread;
		//gettimeofday(&ta,NULL);
		//us_sum+=(ta.tv_sec-tp.tv_sec)*1000000+ta.tv_usec-tp.tv_usec;
	}
	print_counter++;
	prefetch_thread=new std::thread(sample_thr);
	s=(print_counter&0x1)?s2:s1;
	sp=(print_counter&0x1)?sp2:sp1;
	std::vector<Tensor> outputs;
	if(DOUBLE_DQN){
		TF_CHECK_OK(cs->Run({{x,*s}},{y},&outputs));
		q_=outputs[0];
		TF_CHECK_OK(cs->Run({{x,*sp},{qt_x,*sp}},{argmax,qt_y},&outputs));
		ya=outputs[0];
		ynxt=outputs[1];
	}
	else{
		TF_CHECK_OK(cs->Run({{qt_x,*sp},{x,*s}},{y,qt_ymax},&outputs));
		q_=outputs[0];
		ym=outputs[1];
	}
	float r;
	auto qm=q_.matrix<float>();
	if(DOUBLE_DQN){
		auto ynxtm=ynxt.matrix<float>();
		auto yam=ya.flat<int>().data();
		for(int i=0;i<BATCH;i++){
			r=(bat[i].r)?log2tbl[bat[i].r>>2]:0.0;
			if(unlikely(bat[i].isterm)){
				qm(i,bat[i].a)=r;
			}
			else{
				qm(i,bat[i].a)=r+GAMMA*ynxtm(i,yam[i]);
			}
		}
	}
	else{
		auto ymm=ym.flat<float>().data();
		for(int i=0;i<BATCH;i++){
			r=(bat[i].r)?log2tbl[bat[i].r>>2]:0.0;
			if(unlikely(bat[i].isterm)){
				qm(i,bat[i].a)=r;
			}
			else{
				qm(i,bat[i].a)=r+GAMMA*ymm[i];
			}
		}	
	}
		
	TF_CHECK_OK(cs->Run({{x,*s},{yp,q_}},{step},nullptr));
	if(likely(print_counter&0xF))return;
	TF_CHECK_OK(cs->Run({{x,*s},{yp,q_}},{loss},&outputs));
	loss_sum+=outputs[0].scalar<float>()();
	if(unlikely(++train_counter==QT_UPDATE_INT)){
		train_counter=0;
		TF_CHECK_OK(cs->Run({},{qt_update},nullptr));
	}
	if(unlikely(print_counter==PRINT_INT)){
		print_counter=0;
//		std::cout<<us_sum/PRINT_INT<<std::endl;
//		us_sum=0;
//		TF_CHECK_OK(cs->Run({{x,s},{yp,q_}},{y,loss},&outputs));
		//for(int i=0;i<16;i++)std::cout<<bat[BATCH-1].s[i]<<" ";
		//std::cout<<"\n"<<bat[BATCH-1].a<<" "<<bat[BATCH-1].r<<"\n";
		//for(int i=0;i<16;i++)std::cout<<bat[BATCH-1].sp[i]<<" ";
		//std::cout<<bat[BATCH-1].isterm<<std::endl;
//		std::cout<<outputs[1].scalar<float>()<<" "<<q_.matrix<float>()(BATCH-1,bat[BATCH-1].a)<<" "<<outputs[0].matrix<float>()(BATCH-1,bat[BATCH-1].a)<<" "<<r<<std::endl;
//		prefetch_thread=new std::thread(sample_thr);
		std::cout<<loss_sum/(PRINT_INT>>4)<<std::endl;
		loss_sum=0.0;
	}
}

int main(){
	gen_log2();
	dqnagent_init();
	agent_t dqnagent={.getact=dqnagent_getact,.pushmem=dqnagent_pushmem,.train=dqnagent_train};
	srand(time(NULL));
	epsilon=1.0f;
	double epsilon_append=0.98;
	double epsilon_bak;
	while((!replaymem_full)&&replaymem_idx<MEM_START_TRAIN) episode(dqnagent);
	puts("Filled");
	FZTE(j,400){
		unsigned int tscore=0,tstep=0;
		epsilon=1.0/(j*1.5/10.0);
		FZTE(i,100){
			log_t elog=episode(dqnagent);
			tscore+=elog.score;
			tstep+=elog.step;
			//epsilon_append*=0.999f;//7f;
			//epsilon=0.03+epsilon_append;
		}
		printf("Train %d %f %u %u %f\n",j,epsilon,tscore/100,tstep/100,(float)tscore/tstep);
		if(!(j%4)){
			epsilon_bak=epsilon;
			epsilon=0.0f;
			block_train=1;
			tscore=0;
			tstep=0;
			FZTE(i,100){
				log_t elog=episode(dqnagent);
				tscore+=elog.score;
				tstep+=elog.step;
			}
			printf("Test %d %f %u %u %f\n",j,epsilon,tscore/100,tstep/100,(float)tscore/tstep);
			block_train=0;
			epsilon=epsilon_bak;
		}
	}
	delete cs;
	return 0;
}
