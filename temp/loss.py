def performance_and_loss(preds, gold):
    print('gold:', gold.shape)
    print('preds:', preds.shape)
    print('Preds actual:', preds)
    loss = compute_loss(preds, gold)
    print('Preds_max full:', preds.max(1))
    pred_max = preds.max(1)[1]
    print('Pred_max:', pred_max)
    print('pred_max shape:', pred_max.shape)
    gold = gold.contiguous().view(-1)
    print('gold:', gold)
    non_pad_mask = gold.ne(PAD)
    print('non pad mask:', non_pad_mask)
    print('non pad mask shape:', non_pad_mask.shape)
    n_correct = pred_max.eq(gold)
    print('n correct:', n_correct)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    print('n_correct masked:', n_correct)
    return loss, n_correct


def compute_loss(preds, gold):
    print('Gold actual:')
    print(gold)
    gold = gold.contiguous().view(-1)
    print('gold contiguous:', gold.shape)
    print(gold)
    loss = F.cross_entropy(preds, gold, ignore_index=PAD, reduction="sum")
    print('loss:', loss)
    print('loss shape:', loss.shape)
    return loss
