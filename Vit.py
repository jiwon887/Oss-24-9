import os
import time
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from timm.models.layers import trunc_normal_
from torchvision.datasets.cifar import CIFAR10

class EmbeddingLayer(nn.Module):
    def __init__(self, in_chans, embed_dim, img_size, patch_size):
        super().__init__()
        # 32 * 32 사진을 이용 (4*4 filter로 64개 token 생성)
        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_chans, embed_dim, kernel_size= patch_size, stride= patch_size)
        
        # CLS 토큰 생성
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_tokens += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim)) # 위치 임베딩
        
        
        # 토큰 초기화(보편적으로 아래처럼 사용)
        nn.init.normal_(self.cls_token, std= 1e-6)
        trunc_normal_(self.pos_embed, std = 0.2)
    
    
    def forward(self, x):
        B, C, H, W = x.shape        # 배치, 채널수, 높이, 너비
        embedding = self.project(x)     # 입력 이미지를 패치 단위로 나누어 임베딩 벡터로 변환
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)      # 임베딩 벡터를 토큰 형태로 변환
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)      # 클래스 토큰을 배치 크기만큼 확장
        z = torch.cat([cls_tokens, z], dim= 1)      # 클래스 토큰을 배치 크기로 확장

        z = z + self.pos_embed      # 위치 임베딩 벡터를 추가해서 반환
        
        return z
    
    
class MSA(nn.Module):
    def __init__(self, dim = 192, num_heads = 12, qkv_bias = False, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        
        # 입력 차원이 헤드 수로 나누어 떨어져야하는 조건
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # Query, Key, Value를 생성하기 위해 차원을 3배로 늘려서 계산
        self.attn_drop = nn.Dropout(attn_drop)  
        self.proj = nn.Linear(dim, dim)     # attention 결과를 다시 선형변환하여 입력 차원으로 만들어줌
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape       # 배치 크기, 시퀀스 길이, 채널 차원
        
        # 텐서의 차원을 재배열
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Query, Key, Value를 분리
        q,k,v = qkv.unbind(0)
        
        # Query와 Key를 행렬 곱을 통해 attetion score 계산
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)
        
        # Value 벡터와 attention score를 곱하여 attention 결과 계산, 결과 차원으로 재배치
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)        # 원래 차원으로 변환
        x = self.proj_drop(x)
        
        return x
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_feature = in_features   # 출력차원과 입력차원을 동일하게 설정
        
        # fc1 = 첫 번째 선형 레이어, 입력 데이터를 hidden 차원으로 변경
        # act1 = 활성함수, 비선형적으로 변경
        # fc2 = 두 번째 선형 레이어, out_feature 차원으로 다시 변경
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_feature, bias=bias)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x

# 실제 encorder동작
# dim : 입력 차원
# num_heads : MSA 헤드 수
# mlp_ratio : MLP 레이어 차원, 은닉층은 dim * mlp_ratio
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4. , qkv_bias=False, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                        proj_drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
    
class Vit(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12,
                 num_heads=12, mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
        
        self.patch_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size)
        self.blocks = nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio= mlp_ratio, qkv_bias= qkv_bias,
                                            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer= act_layer)
                                      for i in range(depth)])
        
        
        self.norm = norm_layer(embed_dim)
        
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)[:,0]
            
        return x
        
def main():
    
    # 인자파싱, 학습에 필요한 설정 값 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--root', type=str, default='./CIFAR10')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--name', type=str, default='vit_cifar10')
    parser.add_argument('--rank', type=int, default=0)
    ops = parser.parse_args()
    
    device = torch.device('dml' if torch.cuda.is_available() else 'cpu')
    
    #Compose : transform을 연속적으로 실행하여 이미지 데이터 전처리
    #RandomCrop : 이미지를 무작위로 잘라냄
    #RandomHorizontalFlip : 이미지를 좌우 대칭으로 뒤집음, 대칭된 이미지에 대한 일반화
    #ToTensor : 이미지를 Pytorch텐서로 변환
    #Normalize : 이미지의 각 채널을 정규화 (RGB), CIFAR 데이터셋 통계 값이라함 그냥 하라는대로 해
    
    transform_cifar = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4495),
                      std=(0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform_cifar = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=(0.4914, 0.4822, 0.4495),
                                                      std=(0.2023, 0.1994, 0.2010)),
                                        ])
    
    
    # train_set, test_set : 데이터 셋
    # train_loader, test_loader : 데이터 셋에서 배치 단위로 데이터를 로드해주는 도구
    train_set = CIFAR10(root=ops.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)
    
    test_set = CIFAR10(root=ops.root,
                       train=False,
                       download=True,
                       transform=transform_cifar)
    
    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=ops.batch_size)
    
    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=ops.batch_size)
    
    
    #모델
    model = Vit().to(device)
    #손실함수
    criterion = nn.CrossEntropyLoss()
    #옵티마이저
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ops.lr,
                                 weight_decay=5e-5)
    #스케쥴러 : 학습률을 줄이기 위해 사용 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=1e-5)
    
    os.makedirs(ops.log_dir, exist_ok=True)
    
    # 10. ** training **
    print("training...")
    for epoch in range(ops.epoch):

        model.train()
        tic = time.time()
        for idx, (img, target) in enumerate(train_loader):
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]
            # output, attn_mask = model(img, True)
            output = model(img)  # [N, 10]
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if idx % ops.step_size == 0:
                print('Epoch : {}\t'
                      'step : [{}/{}]\t'
                      'loss : {}\t'
                      'lr   : {}\t'
                      'time   {}\t'
                      .format(epoch,
                              idx, len(train_loader),
                              loss,
                              lr,
                              time.time() - tic))

        # save
        save_path = os.path.join(ops.log_dir, ops.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

        torch.save(checkpoint, os.path.join(save_path, ops.name + '.{}.pth.tar'.format(epoch)))

        # 10. ** test **
        print('Validation of epoch [{}]'.format(epoch))
        model.eval()
        correct = 0
        val_avg_loss = 0
        total = 0
        with torch.no_grad():

            for idx, (img, target) in enumerate(test_loader):
                model.eval()
                img = img.to(device)  # [N, 3, 32, 32]
                target = target.to(device)  # [N]
                output = model(img)  # [N, 10]
                loss = criterion(output, target)

                output = torch.softmax(output, dim=1)
                # first eval
                pred, idx_ = output.max(-1)
                correct += torch.eq(target, idx_).sum().item()
                total += target.size(0)
                val_avg_loss += loss.item()

        print('Epoch {} test : '.format(epoch))
        accuracy = correct / total
        print("accuracy : {:.4f}%".format(accuracy * 100.))

        val_avg_loss = val_avg_loss / len(test_loader)
        print("avg_loss : {:.4f}".format(val_avg_loss))

        scheduler.step()


if __name__ == '__main__':
    main()    