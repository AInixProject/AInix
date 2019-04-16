def paired_files_to_tsv(
    left_file: str,
    right_file: str,
    out_tsv: str,
    delemiter: str = "\t"
):
    with open(left_file) as lf, \
            open(right_file) as rf, \
            open(out_tsv, "w") as o:
        for l, r in zip(lf, rf):
            o.write(l.strip() + delemiter + r)


if __name__ == "__main__":
    paired_files_to_tsv(
        "../opennmt/data_train_x.txt",
        "../opennmt/data_train_y.txt",
        "./data/data_train.tsv",
    )
    paired_files_to_tsv(
        "../opennmt/data_val_x.txt",
        "../opennmt/data_val_y.txt",
        "./data/data_val.tsv",
    )
