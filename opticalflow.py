# Moore neighborhood
Q = 3

def kl(img0, img1):
    It, Iy, Ix = torch.gradient(torch.vstack((img0, img1)))
    It, Iy, Ix = It[0], Iy[0], Ix[0]
    Ix_patches = Ix.unfold(0, 3, 3).unfold(1, 3, 3)
    Iy_patches = Iy.unfold(0, 3, 3).unfold(1, 3, 3)
    It_patches = It.unfold(0, 3, 3).unfold(1, 3, 3)
    
    V = []
    for iy in range(Ix_patches.size(0)):
        COL = []
        for ix in range(Ix_patches.size(1)):
            p_Ix = Ix_patches[iy, ix].ravel()
            p_Iy = Iy_patches[iy, ix].ravel()
            p_It = It_patches[iy, ix].ravel().unsqueeze(1)
            
            A = torch.stack([p_Ix, p_Iy]).T
            b = p_It
            v = torch.linalg.pinv(A) @ b
            
            COL.append(v.squeeze(1))
        COL = torch.stack(COL)
        V.append(COL)
    return torch.stack(V)

N = 5
pyramid = [
    kl(
        VF.resize(img0, (img0.size(1)//r, img0.size(2)//r)),
        VF.resize(img1, (img1.size(1)//r, img1.size(2)//r))
    ).permute(2, 0, 1) * r * 3
for r in reversed([2 ** i for i in range(N)])]