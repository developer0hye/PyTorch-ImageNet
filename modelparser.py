import argparse
import torch
from pathlib import Path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Model Parser')
parser.add_argument('--pretrained-weights', required=True, type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)

        model_state_dict = checkpoint['model_state_dict']

        top1_accuracy = checkpoint['top1_accuracy']
        best_top1_accuracy = checkpoint['best_top1_accuracy'] * 100.
        best_top1_accuracy = f'{best_top1_accuracy: .2f}'
        best_top1_accuracy = best_top1_accuracy.replace(' ', '')

        print(f"best_top1_accuracy(%): ", best_top1_accuracy)

        model_name = Path(args.pretrained_weights).resolve().stem
        torch.save(model_state_dict, model_name + f"_top1acc{best_top1_accuracy}.pth")