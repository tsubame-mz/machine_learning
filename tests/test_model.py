from context import model

import torch


def test_pol_net():
    in_features, out_features, h1, h2 = 4, 2, 8, 6
    pol = model.PolNet(in_features, out_features, h1, h2)
    print(pol)
    batch_size = 8
    x = torch.randn(batch_size, in_features)
    print(x)
    out = pol(x)
    print(out)


def test_val_net():
    in_features, h1, h2 = 4, 8, 6
    val = model.ValNet(in_features, h1, h2)
    print(val)
    batch_size = 8
    x = torch.randn(batch_size, in_features)
    print(x)
    out = val(x)
    print(out)


def main():
    test_pol_net()
    test_val_net()


if __name__ == "__main__":
    main()
