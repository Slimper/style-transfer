import torch
import torch.optim as optim
from models.vgg import VGGFeatures
from src.utils import load_image, gram_matrix, imshow


def main(content_path, style_path, output_path, device='cuda'):
    content = load_image(content_path).to(device)
    style = load_image(style_path, shape=content.shape[-2:]).to(device)
    generated = content.clone().requires_grad_(True)

    vgg = VGGFeatures().to(device).eval()
    optimizer = optim.LBFGS([generated])

    style_features = vgg(style)
    content_features = vgg(content)
    style_grams = [gram_matrix(f) for f in style_features]

    style_weight = 1e6
    content_weight = 1e0

    run = [0]
    while run[0] <= 300:
        def closure():
            optimizer.zero_grad()
            gen_features = vgg(generated)
            content_loss = torch.nn.functional.mse_loss(gen_features[2], content_features[2])
            style_loss = 0
            for gf, sg in zip(gen_features, style_grams):
                gm = gram_matrix(gf)
                style_loss += torch.nn.functional.mse_loss(gm, sg)
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}, Loss: {total_loss.item()}")
                imshow(generated, title=f"Step {run[0]}")
            return total_loss
        optimizer.step(closure)
    # Save output
    out_img = generated.cpu().clone().squeeze(0)
    out_img = torch.clamp(out_img, 0, 1)
    from torchvision.utils import save_image
    save_image(out_img, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/result.png')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args.content, args.style, args.output, args.device)
