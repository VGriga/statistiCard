import numpy as np
def calculator(players_cards:list, others_cards:list, gone_cards:list) -> (np.float32,np.float32,np.float32):
    '''
    returns: 
        prob to overtake
        prob to blackjack
        prob to undertake
    '''
    inds = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    deck = np.ones(len(inds),dtype=np.uint8)*4
    for card in players_cards+others_cards+gone_cards:
        value = card[:-1]
        deck[inds.index(value)]-=1
    # print('left card =', np.sum(deck))
    # print(Counter(gone_cards))
        
    player_has_ace = 0
    player_count = 0
    
    for card in players_cards:
        value = card[:-1]
        if str.isdigit(value):
            player_count += int(value)
        elif value != 'A':
            player_count += 10
        elif value == 'A':
            player_has_ace += 1
            player_count += 11
            
    if (len(players_cards)==2 and player_has_ace==2) or player_count == 21:
        return -2, -2, -2
    
    if player_count>21:
        return -1, -1, -1
    
    need_to_take = 21 - player_count
    

    if need_to_take < 10 and need_to_take > 1:
        prob_blackjack = deck[inds.index(str(need_to_take))]/np.sum(deck)
        prob_to_overtake = np.sum(deck[inds.index(str(need_to_take))+1:-1])/np.sum(deck)
        prob_to_undertake = (np.sum(deck[:inds.index(str(need_to_take))])+deck[-1])/np.sum(deck)
        
    elif need_to_take == 10:
        prob_blackjack = np.sum(deck[inds.index('10'):inds.index('K')+1])/np.sum(deck)
        prob_to_overtake = 0
        prob_to_undertake = (np.sum(deck[:inds.index(str(10))])+deck[-1])/np.sum(deck)
        
    elif need_to_take == 11:
        prob_blackjack = deck[inds.index('A')]/np.sum(deck)
        prob_to_overtake = 0
        prob_to_undertake = np.sum(deck[:-1])/np.sum(deck)
        
    elif need_to_take == 1:
        prob_blackjack = deck[inds.index('A')]/np.sum(deck)
        prob_to_overtake = np.sum(deck[:-1])/np.sum(deck)
        prob_to_undertake = 0 
    
    else:
        prob_blackjack = 0
        prob_to_overtake = 0
        prob_to_undertake = 1
        
    return  prob_to_overtake, prob_blackjack, prob_to_undertake
        
        
