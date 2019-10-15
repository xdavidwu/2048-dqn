#ifndef __2048_H__
#define __2048_H__
#ifdef __cplusplus
extern "C" {
#endif
	
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOC(x,y)	((x)*4+(y))
#define FZTE(i,n)	for(int i=0;i<n;i++)
#define FATZ(i,n)	for(int i=n;i>=0;i--)
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

typedef unsigned int cell_t;
typedef unsigned int score_t;
typedef unsigned int u32;

typedef enum{
	L=0,
	B=1,
	R=2,
	T=3,
	X=4 // exit
}dir_t;

typedef struct{
	dir_t (*getact)(cell_t *s);
	void (*pushmem)(cell_t *s,dir_t a,score_t r,cell_t *sp,int isterm);
	void (*train)();
}agent_t;

typedef struct{
	score_t score;
	u32 step;
}log_t;

extern dir_t randomagent_getact();


extern log_t episode(agent_t agent);
#ifdef __cplusplus
}
#endif 
#endif
