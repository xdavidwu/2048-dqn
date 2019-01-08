#include "2048.h"

const cell_t init_grid[]={
	0,1,0,0,
	0,2,0,1,
	0,0,0,0,
	1,0,1,2
};

cell_t *new_grid(){
	cell_t *g=malloc(sizeof(cell_t)*16);
	/* 0, 1, 0, 0
	 * 0, 2, 0, 1
	 * 0, 0, 0, 0
	 * 1, 0, 1, 2
	 */
	memcpy(g,init_grid,sizeof(cell_t)*16);
	return g;
}

void print_grid(cell_t *grid){
	FZTE(i,4){
		FZTE(j,4){
			if(grid[LOC(i,j)])printf("\t%u",1<<grid[LOC(i,j)]);
			else printf("\t0");
		}
		puts("");
	}
}

cell_t *move(cell_t *grid,dir_t dir,score_t *score){
	cell_t tmp[4];
	switch(dir){
		case L:
			FZTE(i,4){
				memset(tmp,0,sizeof(cell_t)*4);
				int tmp_idx=0;
				FZTE(j,4){
					if(grid[LOC(i,j)]){
						tmp[tmp_idx++]=grid[LOC(i,j)];
					}
				}
				FZTE(j,tmp_idx-1){ // no leading zeros
					if(tmp[j]==tmp[j+1]){
						tmp[j]+=1;
						*score+=1u<<(tmp[j]);
						tmp[j+1]=0;
					}
				}
				int grid_idx=0;
				for(int a=0;a<tmp_idx;a++){
					if(tmp[a]) grid[LOC(i,grid_idx++)]=tmp[a];
				}
				for(int a=grid_idx;a<4;a++) grid[LOC(i,a)]=0;
			}
			break;
		case B:
			FZTE(i,4){
				memset(tmp,0,sizeof(cell_t)*4);
				int tmp_idx=0;
				FZTE(j,4){
					if(grid[LOC(j,i)]){
						tmp[tmp_idx++]=grid[LOC(j,i)];
					}
				}
				FZTE(j,tmp_idx-1){ // no leading zeros
					if(tmp[j]==tmp[j+1]){
						tmp[j]+=1;
						*score+=1<<(tmp[j]);
						tmp[j+1]=0;
					}
				}
				int grid_idx=0;
				for(int a=0;a<tmp_idx;a++){
					if(tmp[a]) grid[LOC(grid_idx++,i)]=tmp[a];
				}
				for(int a=grid_idx;a<4;a++) grid[LOC(a,i)]=0;
			}
			break;
		case R:
			FZTE(i,4){
				memset(tmp,0,sizeof(cell_t)*4);
				int tmp_idx=3;
				FATZ(j,3){
					if(grid[LOC(i,j)]){
						tmp[tmp_idx--]=grid[LOC(i,j)];
					}
				}
				for(int j=3;j>tmp_idx+1;j--){ // no leading zeros
					if(tmp[j]==tmp[j-1]){
						tmp[j]+=1;
						*score+=1<<(tmp[j]);
						tmp[j-1]=0;
					}
				}
				int grid_idx=3;
				for(int a=3;a>tmp_idx;a--){
					if(tmp[a]) grid[LOC(i,grid_idx--)]=tmp[a];
				}
				for(int a=grid_idx;a>=0;a--) grid[LOC(i,a)]=0;
			}
			break;
		case T:
			FZTE(i,4){
				memset(tmp,0,sizeof(cell_t)*4);
				int tmp_idx=3;
				FATZ(j,3){
					if(grid[LOC(j,i)]){
						tmp[tmp_idx--]=grid[LOC(j,i)];
					}
				}
				for(int j=3;j>tmp_idx+1;j--){ // no leading zeros
					if(tmp[j]==tmp[j-1]){
						tmp[j]+=1;
						*score+=1<<(tmp[j]);
						tmp[j-1]=0;
					}
				}
				int grid_idx=3;
				for(int a=3;a>tmp_idx;a--){
					if(tmp[a]) grid[LOC(grid_idx--,i)]=tmp[a];
				}
				for(int a=grid_idx;a>=0;a--) grid[LOC(a,i)]=0;
			}
			break;
		}
	return grid;
}

int add_cell(cell_t *grid){
	u32 c=0;
	FZTE(i,16) if(!grid[i])c++;
	if(!c)return 0;
	int d=rand()%c;
	cell_t val=(rand()%10)?1:2;
	FZTE(i,16) if(!grid[i]){
		c--;
		if(c==d){
			grid[i]=val;
			return 1;
		}
	}
	return 0;
}

dir_t randomagent_getact(){
	return rand()%4;
}

log_t episode(agent_t agent){
	log_t log;
	log.score=0;
	log.step=0;
	cell_t *grid=new_grid();
	//print_grid(grid);
	while(1){
		score_t os=log.score;
		cell_t og[16];
		memcpy(og,grid,sizeof(cell_t)*16);
		dir_t act=agent.getact(grid);
		move(grid,act,&(log.score));
		log.step++;
		/*char c;
		scanf("%c",&c);
		switch(c){
			case 'l':
				move(grid,L,&s);
				break;
			case 'b':
				move(grid,B,&s);
				break;
			case 'r':
				move(grid,R,&s);
				break;
			case 't':
				move(grid,T,&s);
				break;
			case 'x':
				return 0;
			default:
				continue;
		}*/
		if(!add_cell(grid)){
			agent.pushmem(og,act,log.score-os,grid,1);
			agent.train();
			return log;
		}
		agent.pushmem(og,act,log.score-os,grid,0);
		agent.train();
		//printf("%u\n",s);
		//print_grid(grid);
	}
	return log;
}
