s= tf('s')
sys = 1/s^2
d_int = ss(c2d(sys,1/50))
C = d_int.C
A = d_int.A
B = d_int.B
a = C(1)
T = [1/2/a 1 ; 1/2/a -1]
An = inv(T)*A*T
Bn=inv(T)*B
Cn = [1 0]
newsys = ss(An,Bn,[1 0],0,1/50)
newsys.B
newsys.A

%%
A = [0 1 ; 1 -1]
B = [0 ; 1]
Q = [10 0 ; 0 0]
R = 0.1
SYS = ss(A,B,eye(2),0)
[K,S,E] = lqr(SYS,Q,R,0)
