"""Practical 5"""
import re
from enum import Enum

import simplejson as json

from .nlp import normalize

digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")


def prepare_slot_values_independent():
    """
    Returns:
        delex_list: list
    """
    domains = ['restaurant']
    delex_list = []

    # TODO TASK d)
    # Placeholders, do not remove this.
    delex_area = []
    delex_food = []
    delex_price = []
    class Slots(Enum):
        AREA = "area"
        FOOD = "food"
        PRICERANGE = "pricerange"
        NAME = "name"
        PHONE = "phone"
        ADDRESS = "address"
        POSTCODE = "postcode"

    # read databases
    for domain in domains:
        fin = open('db/' + domain + '_db.json')
        db_json = json.load(fin)
        fin.close()

        for ent in db_json:
            for key, val in list(ent.items()):
                if val == '?' or val == 'free':
                    pass
                elif key == Slots.ADDRESS.value:
                    delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    if "road" in val:
                        val = val.replace("road", "rd")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif "rd" in val:
                        val = val.replace("rd", "road")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif "st" in val:
                        val = val.replace("st", "street")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif "street" in val:
                        val = val.replace("street", "st")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                elif key == Slots.NAME.value:
                    delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    if "b & b" in val:
                        val = val.replace("b & b", "bed and breakfast")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif "bed and breakfast" in val:
                        val = val.replace("bed and breakfast", "b & b")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif "hotel" in val and 'gonville' not in val:
                        val = val.replace("hotel", "")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif "restaurant" in val:
                        val = val.replace("restaurant", "")
                        delex_list.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                elif key == Slots.POSTCODE.value:
                    delex_list.append((normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                elif key == Slots.PHONE.value:
                    delex_list.append((val, '[' + domain + '_' + 'phone' + ']'))

                # TODO TASK d)
                # Add delexicalized tokens for three slots key: area, food and pricerange
                # to the list of all possible dictionary slot-value pairs.
                # The tokens should have the form '[value_name_of_the_slot]'
                # and the delexicalized lists are formed of tuples:
                # (value, [value_name_of_the_slot])
                # Remember to normalize the value before adding it to the list!
                # YOUR CODE HERE:
                elif key == Slots.AREA.value:
                    delex_area.append((normalize(val), '[value_' + 'area' + ']'))
                elif key == Slots.FOOD.value:
                    delex_food.append((normalize(val), '[value_' + 'food' + ']'))
                elif key == Slots.PRICERANGE.value:
                    delex_price.append((normalize(val), '[value_' + 'pricerange' + ']'))
                # YOUR CODE ENDS HERE.

                else:
                    pass

    # more general values add at the end
    delex_list.extend(delex_area)
    delex_list.extend(delex_food)
    delex_list.extend(delex_price)

    return delex_list


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]

    return utt


def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key == domain or key == 'value':
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
            utt = utt[1:-1]

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]

    return utt


if __name__ == '__main__':
    prepare_slot_values_independent()
