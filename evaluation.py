import json
import argparse


def evaluate_dataset_case_level(data):

    tot_cases = 0
    case_macc = 0
    case_hacc = 0

    tot_list = [0, 0, 0]
    macc_list = [0, 0, 0]
    hacc_list = [0, 0, 0]

    for case in data:
        tot_cases += 1

        case_macc_hit = False
        case_hacc_hit = False

        hop_types_in_case = set()

        for item in case["multi_hop_questions"]:
            hop = item["number_of_hops"]
            if hop == 2:
                idx = 0
            elif hop == 3:
                idx = 1
            else:
                idx = 2

            hop_types_in_case.add(idx)

            gold_final = item["multi_hop_question"]["answer"].lower()
            gold_alias = [a.lower() for a in item["multi_hop_question"].get("answer_alias", [])]
            gold_path = [x.lower() for x in item["path"]]

            #
            pred_final = item["final_answer"].lower()
            if not pred_final:
                continue
            final_correct = (
                pred_final == gold_final or gold_final in pred_final or any(pred_final in alias for alias in gold_alias)
            )

            if final_correct:
                case_macc_hit = True

            hop_correct = True
            single_hops = item["single_hop_questions"]

            if len(single_hops) == hop:

                for shq, gold in zip(single_hops, gold_path):

                    gold_main = gold.lower()
                    alias_list = [a.lower() for a in shq.get("answer_alias", [])]
                    gold_set = set([gold_main] + alias_list)

                    pred = shq["answer"].strip().lower()

                    if pred not in gold_set:

                        hop_correct = False
                        break
            else:
                hop_correct = False

            if hop_correct:

                case_hacc_hit = True

        for idx in hop_types_in_case:
            tot_list[idx] += 1

        if case_macc_hit:
            case_macc += 1
            for idx in hop_types_in_case:
                macc_list[idx] += 1

        if case_hacc_hit:
            case_hacc += 1
            for idx in hop_types_in_case:
                hacc_list[idx] += 1

    print("========== Case-Level Result ==========")
    print(f"M-Acc = {case_macc}/{tot_cases} = {case_macc/tot_cases:.4f}")
    print(f"H-Acc = {case_hacc}/{tot_cases} = {case_hacc/tot_cases:.4f}")

    print("\n===== Breakdown by Hop Count (Case-Level) =====")
    print(f"2-hop: case_tot={tot_list[0]}, case_m-acc={macc_list[0]}, case_h-acc={hacc_list[0]}")
    print(f"3-hop: case_tot={tot_list[1]}, case_m-acc={macc_list[1]}, case_h-acc={hacc_list[1]}")
    print(f"4-hop: case_tot={tot_list[2]}, case_m-acc={macc_list[2]}, case_h-acc={hacc_list[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluate_dataset_case_level(data)
