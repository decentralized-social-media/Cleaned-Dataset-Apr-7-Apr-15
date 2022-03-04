from __future__ import print_function
import sys
import time
from datetime import timedelta
import io
from beem import Steem
from beem.blockchain import Blockchain
from beem.instance import shared_blockchain_instance
from beem.account import Account
import pandas as pd
#get data from steemit api
stm = Steem("https://api.steemit.com")

#create data frame for the data
dataset_columnlist =['parent author', 'parent permlink',
                             'author','permlink','title','body','json metadata',
                             'timestamp','block num','profile json metadata','profile posting json metadata']
dataset_collection = pd.DataFrame(columns = dataset_columnlist)

#class function for extract desired information in the single comment and its account
class basic_comment_info(object):
    def comment(self, comment_piece):
        account = comment_piece['author']
        user_account = Account(account, steem_instance=stm)
        data = [comment_piece['parent_author'],
                comment_piece['parent_permlink'],
                comment_piece['author'],
                comment_piece['permlink'],
                comment_piece['title'],
                comment_piece['body'],
                comment_piece['json_metadata'],
                comment_piece['timestamp'].isoformat(),
                comment_piece['block_num'],
                user_account.json()['json_metadata'],
                user_account.json()['posting_json_metadata']]
        dataset_collection.loc[len(dataset_collection.index)+1] = data
        
if __name__ == "__main__":
    get_info = basic_comment_info()
    blockchain = Blockchain(blockchain_instance=stm)
    duration_s = 60 * 60 * 24
    blocksperday = int(duration_s / 3)
    #sample for stream today's data
    current_block_num = blockchain.get_current_block_num()
    last_block_id = current_block_num + blocksperday
    #can change the number of blocks to the desired date to extract
    for vote in blockchain.stream(start =  current_block_num, stop = last_block_id,opNames=["comment"]):
        get_info.comment(vote)
    dataset_collection.to_excel('winterolympics.xlsx')
