## Error install ros
Ubuntu 18.04 gpg dirmngr IPC connect call failed

Can try this:

- use sudo apt update then will get something like The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 5523BAEEB01FA116
- Got the pubkey 5523BAEEB01FA116
- Use curl -sL "http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x5523BAEEB01FA116" | sudo apt-key add Replace 0x5523BAEEB01FA116 with 0x<your pubkey in 2step>
- complete, you can run sudo apt udpate no error
