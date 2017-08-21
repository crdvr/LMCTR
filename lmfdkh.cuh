
__global__ void cv_(float *a,float *b,int n,int sid,rp2 *rd){
	SETID;
	//int r=rd->r,r_=rd->r_,e=rd->e,cvthn=rd->cvthn;
	//int i=id/r_,j=id%r_,k,s,en;
	int rm=(id+sid)%(rd->U*rd->V),i=rm/U,j=rm%U,k;
	float *slt=rd->sl,du=rd->du,p,q,t;
	if(id<n){
		a[id+sid]=0;

		for(k=j-U+1;k<=j+U-1;k++){
			a[id+sid]+=(k<0||k>=U?0:b[id+sid-j+k])*slt[(k-j>=0?k-j:j-k)];
		}

		a[id+sid]*=du;
	}
}

__global__ void bp_(float *p,float *a,int n,int nth,int sh,int shth,rp2 *r){
	SETID;
	int r_=r->r_,th=r->th,w=r->r,h=r->r;
	float dr=r->dr,rs=r->rs,*c=r->c,*s=r->s;
	float v=0;
	float q,dq,x,y;
	int i,ix=(id+sh)%w,iy=(id+sh)/w;
	if(id<n){
		x=ID(ix,w);
		y=ID(iy,h);

		for(i=0;i<nth;i++){
			v+=(1-(dq=(q=(x*c[i+shth]+y*s[i+shth]-rs)/dr+(r_-1)/2.0)-(int)q))*a[i*r_+(int)q]+dq*a[i*r_+(int)q+1];
		}
		v*=1.0/2.0/th;//1/2pi*pi/th
		p[iy*w+ix-sh]+=v;
	}
}

__global__ void w3_(float *g,int v,int sid,rc3* rd){
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int U=rd->U,V=rd->V,n=U*V;
	int k=(id+sid)%n;
	float du=rd->du,dv=rd->dv,su=rd->su,sv=rd->sv,u=(-(U-1)/2.0+k%U)*du+su,v=(-(V-1)/2.0+k/U)*dv+sv,R=rd->R;
	if(id<v)g[id]=g[id]*R/sqrtf(R*R+u*u+v*v);

}
#define RDBP	read_device(r->cv,r->cvf,r_*OVR(j,bpthn,th),r->r_bpthn*j);\
	CUF(bp_,(r->bp,r->cv,OVR(i,bprcn,rr),OVR(j,bpthn,th),r->bprcn*i,r->bpthn*j,rd));
#define HD printf("%d ",i);SETBN(OVR(i,bprcn,rr));CUF(ia,(r->bp,0,OVR(i,bprcn,rr)));
#define RDCV	printf("%d ",i);SETBN(OVR(i,cvn,N));\
	read_device(r->rd,r->in,OVR(i,cvn,N),cvn*i);\
	CUF(w3_,(r->rd,OVR(i,cvn,N),r->cvn*i,r->rd));\
	CUF(cv_,(r->cv,r->rd,OVR(i,cvn,N),cvn*i,rd));\
	write_device(r->cv,r->cvf,r->r_*OVR(i,cvthn,th));

void rp2_::bp(){
	int i,j,blockN;
	int cvthn=r->cvthn,bpthn=r->bpthn,bprcn=r->bprcn,rr=r->rr,th=r->th,r_=r->r_;
	int N=r->U*r->V*th,cvn=r->cvn;
	printf("CV\n");
	REPREM(i,N,cvn,RDCV);
	printf("\nBP\n");
	REPREM(i,rr,bprcn,HD REPREM(j,th,bpthn,RDBP) write_device(r->bp,r->bpf,OVR(i,bprcn,rr)););
	printf("\n");
}

#define ITS(ind,ax) (dxyz[ax]*(ind-XYZ[ax]/2.0)) //index to space coordinate
#define IEQ(a,b,c) ((a)<=(b)&&(b)<=(c))
#define Xt(t) (c*r-s*(t))
#define Yt(t) (c*(t)+s*r)
#define SIS(i) (i<2?(za=(Zt(t[i])-z0)/dz)+(ya=(Yt(t[i])-y0)/dy)+(xa=(i==1)*X):(i<4?(xa=(Xt(t[i])-x0)/dx)+(ya=(i==3)*Y)+(za=(Zt(t[i])-z0)/dz):(xa=(Xt(t[i])-x0)/dx)+(za=(i==5)*Z)+(ya=(Yt(t[i])-y0)/dy)))
//set intersection
#define ISJ(i) (i<4?(i<2?IEQ(y0,Yt(t[i]),y1)&&IEQ(z0,Zt(t[i]),z1):IEQ(x0,Xt(t[i]),x1)&&IEQ(z0,Zt(t[i]),z1)):IEQ(y0,Yt(t[i]),y1)&&IEQ(x0,Xt(t[i]),x1))
#define ISJa(i) if(ISJ(i)){SIS(i)}else{return 0;}
#define ISJav(i) if(ISJ(i)){(i<2?(ya=(Yt(t[i])-y0)/dy)+(xa=(i==1)*X):(xa=(Xt(t[i])-x0)/dx)+(ya=(i==3)*Y));}else{return ;}

#define ISI(x) (EPC(x-(int)x)?1:(EPC(x-(int)x+1)?1:(EPC(x-(int)x-1)?1:0)))
#define INT(x) (EPC(x-(int)x)?(int)x:(EPC(x-(int)x+1)?(int)x-1:(EPC(x-(int)x-1)?(int)x+1:(int)x)))

//#define MIN(a,b,c) (b<a?(c<b?c:b):(c<a?c:a))
#define MIN2(a,b) (a<b?a:b)
#define MIN3(a,b,c) MIN2(MIN2(a,b),c)
#define MINFP(t,a,b,c,inx,iny,inz) (b<a?(c<b?(inz=1)+(iny=0)+(inx=0)+(t=c):(iny=1)+(inx=0)+(iny=0)+(t=b)):(c<a?(inz=1)+(iny=0)+(inx=0)+(t=c):(inx=1)+(iny=0)+(inz=0)+(t=a)))
#define RNG(x,X) (0<=x&&x<X)
#define BOX tx=10000;ty=10000;tz=10000;if(sx!=0)tx=((nix=ix+(sx==-1?-1*ISI(x):1))-x)/(x1-x0);\
if(sy!=0)ty=((niy=iy+(sy==-1?-1*ISI(y):1))-y)/(y1-y0);if(sz!=0)tz=((niz=iz+(sz==-1?-1*ISI(z):1))-z)/(z1-z0);\
	MINFP(tmin,tx,ty,tz,inx,iny,inz);if(tmin<0){ (inx?(tmin=ty):(tmin=tx)); }/*if(tmin<0){ STOP; }*//*MIN(tx,ty,tz);*/\
	v+=(!RNG(ix,X)||!RNG(iy,Y)||!RNG(iz,Z)?0:a[iz*X*Y+iy*X+ix])*tmin*l;\
	x=(inx?nix:tmin*(x1-x0)+x);y=(iny?niy:tmin*(y1-y0)+y);z=(inz?niz:tmin*(z1-z0)+z);\
	ix=INT(x);iy=INT(y);iz=INT(z);isx=ISI(x);isy=ISI(y);isz=ISI(z);
//x,ix ixはxがある箱の座標　傾きが0かどうか 次の交点を見つける　xが整数なら
#define Q_BOX(x,y,ix,iy) tx=10000;ty=10000;tz=10000;if(sx!=0)tx=((nix=ix+(sx==-1?-1*ISI(x):1))-x)/(x1-x0);\
if(sy!=0)ty=((niy=iy+(sy==-1?-1*ISI(y):1))-y)/(y1-y0);/*if(sz!=0)tz=(iz+(sz==-1?-1*ISI(z):1)-z)/(z1-z0);*/\
	MIN(tmin,tx,ty,inx,iny);if(tmin<0){ (inx?(tmin=ty):(tmin=tx)); }/*if(tmin<0){ STOP; }*//*MIN(tx,ty,tz);*/\
	atomicAdd(a+iy*X+ix,(!RNG(ix,X)||!RNG(iy,Y)?0:v)*tmin*l);\
	x=(inx?nix:tmin*(x1-x0)+x);y=(iny?niy:tmin*(y1-y0)+y);/*z=tmin*(z1-z0)+z;*/\
	ix=INT(x);iy=INT(y);isx=ISI(x);isy=ISI(y);/*iz=INT(z);*/

float fp_array_host(float* a,int X,int Y,int Z,float x0,float x1,float y0,float y1,float z0,float z1){
	float l=sqrtf((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));
	int sx,sy,sz;
	int inx,iny,nix,niy;
	int ix0,iy0,iz0,ix,iy,iz;
	ix0=INT(x0);iy0=INT(y0);iz0=INT(z0);
	//ix1=INT(x1);iy1=INT(y1);iz1=INT(z1);
	//if(!RNG(ix0,X)||!RNG(ix1,X)||!RNG(iy0,Y)||!RNG(iy1,Y)||!RNG(iz0,Z)||!RNG(iz1,Z)){ return 0; }
	sx=2*(x1>x0)-1;sy=2*(y1>y0)-1;sz=2*(z1>z0)-1;
	if(EPC(x1-x0)){ sx=0; }if(EPC(y1-y0)){ sy=0; }if(EPC(z1-z0)){ sz=0; }
	if(0==sx&&0==sy&&0==sz){ return 0; }
	ix=ix0;iy=iy0;iz=iz0;
	float x,y,z,tx,ty,tz,tmin;
	x=x0;y=y0;z=z0;
	int isx,isy,isz;
	float v=0;
	float len=0,len0=0,len1=0,len2=0,len3=0,t0=0,tm1=0,t2=0,t3=0;
	int c=0;
	while(len<1){//len<l
		BOX;
		if(tmin<0){
			x+y+z;
		}
#ifdef FPAH_CHECK
		if(threadIdx.x==128&&threadIdx.y==0&&threadIdx.z==0&&(len+tmin<0||len+tmin>1.5)){
			len=len;
		}
#endif
		len+=tmin;
#ifdef FPAH_CHECK2
		t0=tm1;
		tm1=t2;
		t2=t3;
		t3=tmin;
		len0=len1;
		len1=len2;
		len2=len3;
		len3=len;
		c++;
#endif
	}
	return v;

}
float fp_host_one(float *a,float x0,float x1,float y0,float y1,float z0,float z1,
	float u,float v,int X,int Y,int Z,float c,float s){
	int i,sw,n=1;
	int ind[]={0,1,2,3,4,5};//0:left,1:right,2:bottom,3:top
	int ia,ib;
	float t[4];
	float xa,xb,ya,yb,za,zb,yx,xy,swf;//0<=xa<=X
	if(EPC(u*s+R*c)){
		t[2]=(y0-R*s)*R/(u*c-R*s);
		t[3]=(y1-R*s)*R/(u*c-R*s);
		t[4]=R*z0/v;
		t[5]=R*z1/v;
		while(n>0){
			n=0;
			for(i=2; i<5; i++){
				if(t[ind[i]]>t[ind[i+1]]){ sw=ind[i]; ind[i]=ind[i+1]; ind[i+1]=sw; n++; }
			}
		}
		ia=ind[3];
		ib=ind[4];
	}
	else if(EPC(u*c-R*s)){
		t[0]=(x0-R*c)*R/(-u*s-R*c);
		t[1]=(x1-R*c)*R/(-u*s-R*c);
		t[4]=R*z0/v;
		t[5]=R*z1/v;
		ind[0]=0;
		ind[1]=1;
		ind[2]=4;
		ind[3]=5;

		while(n>0){
			n=0;
			for(i=0;i<3; i++){
				if(t[ind[i]]>t[ind[i+1]]){ sw=ind[i]; ind[i]=ind[i+1]; ind[i+1]=sw; n++; }
			}
		}
		ia=ind[1];
		ib=ind[2];
	}
	else if(EPC(v)){

		t[0]=(x0-R*c)*R/(-u*s-R*c);
		t[1]=(x1-R*c)*R/(-u*s-R*c);
		t[2]=(y0-R*s)*R/(u*c-R*s);
		t[3]=(y1-R*s)*R/(u*c-R*s);
		while(n>0){
			n=0;
			for(i=0;i<3; i++){
				if(t[ind[i]]>t[ind[i+1]]){ sw=ind[i]; ind[i]=ind[i+1]; ind[i+1]=sw; n++; }
			}
		}
		ia=ind[1];
		ib=ind[2];
	}else{//S=-xs+yc r=xc+ys  
		t[0]=(x0-R*c)*R/(-u*s-R*c);
		t[1]=(x1-R*c)*R/(-u*s-R*c);
		t[2]=(y0-R*s)*R/(u*c-R*s);
		t[3]=(y1-R*s)*R/(u*c-R*s);
		t[4]=R*z0/v;
		t[5]=R*z1/v;

		while(n>0){
			n=0;
			for(i=0; i<5; i++){
				if(t[ind[i]]>t[ind[i+1]]){ sw=ind[i]; ind[i]=ind[i+1]; ind[i+1]=sw; n++; }
			}
		}
		ia=ind[2];
		ib=ind[3];
	}

	ISJa(ia);
	SIS(ib);
	//#define RTL
#ifdef RTL
	return (xa-xb)*(xa-xb)+(ya-yb)*(ya-yb)+(za-zb)*(za-zb);
#else
	return fp_array_host(a,X,Y,Z,xa,xb,ya,yb,za,zb);
#endif

#ifdef FP_1_HOST_OLD
	//if(xa > X||xb > X||ya > Y||yb > Y||za > Z||zb > Z){dum(&X);}
	if((xb-xa)*(xb-xa)<=EPS){
		if(yb<ya){
			swf=ya; ya=yb; yb=swf;
			swf=xa; xa=xb; xb=swf;
		}
		xy=0;
#ifdef RTL
		return (xa-xb)*(xa-xb)+(ya-yb)*(ya-yb);//+(za-zb)*(za-zb);
#else
		return fpy2n(a,X,Y,xa,xb,ya,yb,xy);
#endif
	}
	else if((yx=(ya-yb)/(xa-xb))*yx>1.0){
		if(yb<ya){
			swf=ya; ya=yb; yb=swf;
			swf=xa; xa=xb; xb=swf;
		}
		xy=1.0/yx;
#ifdef RTL
		return (xa-xb)*(xa-xb)+(ya-yb)*(ya-yb);//+(za-zb)*(za-zb);
#else
		return fpy2n(a,X,Y,xa,xb,ya,yb,xy);
#endif
	}
	else{
		if(xb<xa){
			swf=ya; ya=yb; yb=swf;
			swf=xa; xa=xb; xb=swf;
		}
		//yx=(ya-yb)/(xa-xb);
#ifdef RTL
		return (xa-xb)*(xa-xb)+(ya-yb)*(ya-yb)+(za-zb)*(za-zb);
#else
		return fpx2n(a,X,Y,xa,xb,ya,yb,yx);
#endif
	}
#endif
}

void fp_host_(float *a,float *pr,float x0,float x1,float y0,float y1,float z0,float z1,
	int X,int Y,int Z,int nth,int sth,rp2 *rd){
	//SETID;
	int i,j;
	int U=rd->U,V=rd->V;
	float du=rd->du,dv=rd->dv,su=rd->su,sv=rd->sv;
	for(int id=0;id<nth*U*V;id++){		
		i=id/(U*V);
		j=id%(U*V);
		k=j/U;
		l=j%U;
		u=du*(l-(U-1)/2.0)+su;
		v=dv*(k-(V-1)/2.0)+sv;
		(id%10000==0?printf("%d ",id):0);
		pr[id]+=fp_host_one(a,x0,x1,y0,y1,z0,z1,X,Y,Z,u,v,c[i+sth],s[i+sth]);
	}
}
//rと角度を引数にしている　発射点と角度 u,v,th
void rp2_::fp_host(){
	int X=r->r,Y=r->r,th=r->th;
	float dx=r->dx,dy=r->dy,dr=r->dr,*c_h,*s_h,*a,*b;
	int blockN,i,j,k,l,fpthn=r->fpthn,nx=r->nx,ny=r->ny;
	char *fn=r->bpf,*fn1=r->fpf;
	FILE *f;
	MAL(a,float,nx*ny);
	MAL(b,float,fpthn*r->r);
	fo(&f,fn1,"wb");
	MAL(c_h,float,th);
	CUCP(c_h,r->c,float,th,D2H);
	MAL(s_h,float,th);
	CUCP(s_h,r->s,float,th,D2H);
#define FHDHOST iah(b,0,OVR(l,fpthn,th)*r->r);
#define RDFPHOST iah(a,-1,X*ny);read_host(a,fn,X*ny,i*X*Y);\
	printf("(%d,%d)",j,k);\
	fp_host_(a,b,ITS(0,0),ITS(X,0),ITS(0,1),ITS(Y,1),ITS(i*nz,2),ITS(OVU(i,nz,Z),2),\
	dx,dy,dz,X,Y,nz,r->r,dr,d_c,d_s,OVR(l,fpthn,th),l*fpthn,r);
#define FFTHOST printf("%d ",l);write_host(b,fn1,OVR(l,fpthn,th)*U*V);
#define FPCONDHOST 1
	REPREM(l,th,fpthn,FHDHOST REPREM(j,Y,ny,REPREM(k,X,nx,if(FPCONDHOST){ RDFPHOST })
		) FFTHOST);
}
//X,ny,1でやる　
//X,Y,nzでやる U,V,fpthnに射影する 

/*
fp_host_(a,b,ITS(0,0),ITS(X,0),ITS(0,1),ITS(ny,1),ITS(i,2),ITS(i+1,2),\
	dx,dy,dz,X,ny,1,r->r,dr,d_c,d_s,OVR(l,fpthn,th),l*fpthn,r);
	*/