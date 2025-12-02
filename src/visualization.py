def format_positions(positions):
    return ['{0: .3f}'.format(x) for x in positions]

def print_loss(epoch, loss, outputs, target, is_train=True, is_debug=False):
    loss_type = "train loss:" if is_train else "valid loss:"
    print("epoch", str(epoch), loss_type, str(loss))
    if is_debug:
        print("example pred:", format_positions(outputs[0].tolist()))
        print("example real:", format_positions(target[0].tolist()))


def print_CILP_results(epoch, loss, logits_per_img, is_train=True):
    if is_train:
        print(f"Epoch {epoch}")
        print(f"Train Loss: {loss} ")
    else:
        print(f"Valid Loss: {loss} ")
    print("Similarity:")
    print(logits_per_img)
