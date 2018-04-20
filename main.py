#!/usr/bin/env python3

import snd_reader as sr


def main():
    train_features = sr.getTrain_TargetFeatures()
    print(train_features)


if __name__ == '__main__':
    main()