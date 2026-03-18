import torch, torch.nn as nn, math
 
class PatchEmbed(nn.Module):
    def __init__(self,img=224,patch=16,ch=3,dim=384):
        super().__init__(); self.n=(img//patch)**2
        self.proj=nn.Conv2d(ch,dim,patch,patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)
 
class MHSA(nn.Module):
    def __init__(self,dim,heads=6):
        super().__init__(); self.h=heads; self.hd=dim//heads; self.s=self.hd**-0.5
        self.qkv=nn.Linear(dim,3*dim); self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        B,N,C=x.shape
        q,k,v=self.qkv(x).reshape(B,N,3,self.h,self.hd).permute(2,0,3,1,4).unbind(0)
        a=torch.softmax((q@k.transpose(-2,-1))*self.s,-1)
        return self.proj((a@v).transpose(1,2).reshape(B,N,C))
 
class TBlock(nn.Module):
    def __init__(self,dim=384,heads=6,mlp_ratio=4):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=MHSA(dim,heads)
        self.n2=nn.LayerNorm(dim); md=int(dim*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(dim,md),nn.GELU(),nn.Linear(md,dim))
    def forward(self,x): return x+self.mlp(self.n2(x+self.attn(self.n1(x))))
 
class ViT(nn.Module):
    def __init__(self,img=224,patch=16,cls=200,dim=384,depth=6,heads=6):
        super().__init__(); self.pe=PatchEmbed(img,patch,3,dim)
        n=self.pe.n; self.cls_tok=nn.Parameter(torch.zeros(1,1,dim))
        self.pos=nn.Parameter(torch.zeros(1,n+1,dim))
      self.blocks=nn.Sequential(*[TBlock(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,cls)
        nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.pe(x)
        x=torch.cat([self.cls_tok.expand(x.size(0),-1,-1),x],1)+self.pos
        return self.head(self.norm(self.blocks(x))[:,0])
 
model=ViT(224,16,200,384,6,6); x=torch.randn(2,3,224,224); out=model(x)
print(f"ViT output: {out.shape} | Params: {sum(p.numel() for p in model.parameters()):,}")
