from __future__ import print_function
import sys
from datetime import timedelta
import time
import io
from beem import Steem
from beem.blockchain import Blockchain
from beem.instance import shared_blockchain_instance
from beem.utils import parse_time
from beem.account import Account
import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import pandas as pd
stm = Steem("https://api.steemit.com")
columnlist =['parent author', 'parent permlink',
                             'author','permlink','title','body','json metadata',
                             'timestamp','block num','profile json metadata','profile posting json metadata']
df1 = pd.DataFrame(columns = columnlist)

class DemoBot(object):
    def comment(self, comment_event):
        account = comment_event['author']
        account1 = Account(account, steem_instance=stm)
        data = [comment_event['parent_author'],
                comment_event['parent_permlink'],
                comment_event['author'],
                comment_event['permlink'],
                comment_event['title'],
                comment_event['body'],
                comment_event['json_metadata'],
                comment_event['timestamp'].isoformat(),
                comment_event['block_num'],
                account1.json()['json_metadata'],
                account1.json()['posting_json_metadata']]
        df1.loc[len(df1.index)+1] = data
        print(data)
if __name__ == "__main__":
    tb = DemoBot()
    blockchain = Blockchain(blockchain_instance=stm)
    duration_s = 60 * 60 * 24
    blocksperday = int(duration_s / 3)
    current_block_num = blockchain.get_current_block_num()
    last_block_id = 55769806 + blocksperday
    for vote in blockchain.stream(start = 55769806, stop = last_block_id,opNames=["comment"]):
        tb.comment(vote)
    df1.to_excel('/Users/catherine/Desktop/biden1.xlsx')
