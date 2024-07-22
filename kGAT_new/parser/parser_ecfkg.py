import argparse


def parse_ecfkg_args():
    parser = argparse.ArgumentParser(description="Run ECFKG.")

    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='amazon-book',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--data_dir', nargs='?', default='./datasets/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='./datasets/pretrain/',
                        help='Path of learned embeddings.')
    #parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/ECFKG/model.pth',
                        #help='Path of stored model.')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity / relation Embedding size.')

    parser.add_argument('--random_embedding', type=float, default=0.0,
                        help='defense: random embedding, larger the value is, stronger the defense. 0-inf')
    parser.add_argument('--random_rec', type=float, default=0.0,
                        help='defense: random recommendation, larger the value is, stronger the defense. 0-1')

    parser.add_argument('--train_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2000,
                        help='Test batch size (the user number to test every batch).')
    parser.add_argument('--topK', type=int, default=100,
                        help='Recommendation K for testing')

    parser.add_argument('--lr', type=float, default=0.5,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=300,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=2,
                        help='Number of epoch for early stopping')

    parser.add_argument('--print_every', type=int, default=100,
                        help='Iter interval of printing loss.')
    parser.add_argument('--evaluate_every', type=int, default=20,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')
    parser.add_argument('--gpu_ind', type=int, default=3,
                        help='index of gpu')
    parser.add_argument('--train_shadow', type=int, default=0,
                        help='train target: 0 or shadow: 1')
    parser.add_argument('--get_output_only', type=int, default=0,
                        help='train model: 0 or only load model and get recommendation: 1')

    parser.add_argument('--partial_option', type=float, default=0.0,
                        help='if not zero, load another partial option (0.05, 0.01)')

    parser.add_argument('--www_baseline', type=int, default=0,
                        help='if not zero, use www baseline')
    parser.add_argument('--www', type=float, default=0,
                        help='if not zero, use www baseline')

    parser.add_argument('--kg_only', type=float, default=0,
                        help='if not zero, use kg only for KMIA+')
    parser.add_argument('--kg_partial', type=float, default=0.0,
                        help='if not zero, load another partial option (0.05, 0.01)')


    args = parser.parse_args()

    if args.train_shadow:
        shadow_sub_name = "shadow({})".format(args.partial_option) if args.partial_option else "shadow"
        save_dir = 'trained_model/ECFKG/{}/{}_embed-dim{}_lr{}_pretrain{}/'.format(
            args.data_name, shadow_sub_name, args.embed_dim, args.lr, args.use_pretrain)
    else:
        save_dir = 'trained_model/ECFKG/{}/{}_embed-dim{}_lr{}_pretrain{}/'.format(
            args.data_name, 'target', args.embed_dim, args.lr, args.use_pretrain)

        if args.www > 0:
            save_dir = save_dir.replace("target", f"WWW_{args.www}_target")
    args.save_dir = save_dir
    pretrain_model_path = save_dir + "model.pth"
    args.pretrain_model_path = pretrain_model_path
    return args


