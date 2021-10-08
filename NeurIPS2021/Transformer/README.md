# 记录阅读Transoformer的一些体会,仅供自己大四加深对Transformer的理解

# Contents
- [2021.10.8](#2021.10.8)

## [2021.10.8](#Contents)

### [Transformer in Transformer](https://arxiv.org/abs/2103.00112)
&nbsp;&nbsp;&nbsp;&nbsp;文章引言就非常有意思，不知道是不是上传的时候就是已经注明有MindSpore的框架的(如果是那么一看就知道是诺亚方舟实验室的作品了),具体的代码可以参考[**码云**](https://gitee.com/mindspore/models/tree/master/research/cv/TNT)和[**Github**](https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch)

#### [知乎大佬的论文解读](https://zhuanlan.zhihu.com/p/355848545)

```python

# 首先不得不佩服Ross Wightman的timm库,助力极简的算法开发.这里我们先看一下部分的源码实现,以便在后面可以更好的阅读论文的相关内容

    ...
    self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
    self.outer_pos = nn.Parameter(torch.zeros(1, num_patches + 1, outer_dim))
    self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))    
    ...
# 第一个重点,作者在这里使用了内部和外部的位置编码
class Block(nn.Module):
    """ TNT Block
    """
    ...
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
# 第二个重点,作者用inner_dim参数控制了Block模块,这里也就是TNT的实现过程,主要看两个地方就可以,其他都是基本的imagenet train的pipline
```

<div align=center>
<img src="Transformer in Transformer/image/inner_block.png">
</div>

&nbsp;&nbsp;&nbsp;&nbsp;其实所谓的inner_block就是对图片以patch的形式进行同样的vit的流程

&nbsp;&nbsp;&nbsp;&nbsp;比如输入是
$$[BatchSize, 3, 224, 224]$$
可以得到其$patch\_size=16$的张量为
$$[BatchSize, \frac{224}{16}\times\frac{224}{16},3\times16\times16]$$

&nbsp;&nbsp;&nbsp;&nbsp;前者为outer_token,后者为inner_token,对两者进行分别的vit运算就是所谓的Transformer in Transformer

```python
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C
        # print("self.inner_pos", self.inner_pos.shape)  # ([1, 16, 40])
        # print("inner_tokens", inner_tokens.shape)

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)
'''
    在Block函数中,作者就对两个token进行了分别的定义运算
    具体可以看到inner_token和outer_token之间存在一个加法的交互
    Block通过has_inner操作(其实全部的block都是has_inner=True的)
'''
```













