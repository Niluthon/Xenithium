// xenithium-bench: the short prompts, by category, in two kinds (PromptSet) measured together (measured_prompts).
// A category is a kind of output with its own token statistics (code repeats itself and drafts well, prose does not),
// so the decode rate is reported per category and combined by a weight mix (weight_mixes). The texts are this
// project's own. Changing one changes every later decode number: bump kMethodVersion (corpus.hpp).
#include "corpus.hpp"

#include <array>
#include <vector>

namespace xenithium::bench {
namespace {

constexpr int kChatTokens = 250, kCodeTokens = 600, kReasoningTokens = 600, kMathTokens = 600, kProseTokens = 800, kJsonTokens = 300, kEditTokens = 500,
              kSummaryTokens = 200;

constexpr std::array kPrompts = {
    ShortPrompt{"chat-followup", "chat",
                "I started running three times a week about a month ago and my knees ache a little after the longer runs. "
                "Is that normal for a beginner, and what should I change first?",
                kChatTokens},
    ShortPrompt{"chat-recommend", "chat",
                "I have a free weekend and about 200 euros. I like hiking, old towns and good coffee, and I live in Munich. "
                "Suggest two different plans and tell me which one you would pick.",
                kChatTokens},
    ShortPrompt{"chat-explain", "chat", "Why does the moon look bigger near the horizon? Explain it the way you would to a curious twelve-year-old.", kChatTokens},

    ShortPrompt{"code-merge-intervals", "code",
                "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals given as (start, end) "
                "tuples in any order. Include a docstring, type hints and five unit tests with pytest.",
                kCodeTokens},
    ShortPrompt{"code-lru-cache", "code",
                "Implement a thread-safe LRU cache in C++20 as a class template LruCache<Key, Value> with get, put and a capacity "
                "given at construction. Use std::list and std::unordered_map, and explain the complexity of each operation in comments.",
                kCodeTokens},
    ShortPrompt{"code-rate-limiter", "code",
                "Write a token-bucket rate limiter in Go: a type Limiter with Allow() bool and Wait(ctx context.Context) error, "
                "safe for concurrent use, with a table-driven test.",
                kCodeTokens},

    ShortPrompt{"reason-schedule", "reasoning",
                "Five talks (A to E) must be scheduled in five consecutive one-hour slots. A must be before C. B cannot be first or "
                "last. D must be directly after E. C cannot be directly after A. List every valid order and show how you ruled the others out.",
                kReasoningTokens},
    ShortPrompt{"reason-estimate", "reasoning",
                "Estimate how many litres of water a city of 500,000 people uses for showers in one year. State every assumption, "
                "compute step by step, and give a low and a high bound.",
                kReasoningTokens},
    ShortPrompt{"reason-plan", "reasoning",
                "A team of four has six weeks to move a web service from one data centre to another with no more than ten minutes "
                "of downtime. Lay out a plan week by week, name the three biggest risks and how each is reduced.",
                kReasoningTokens},

    ShortPrompt{"math-derivative", "math",
                "Differentiate f(x) = x^3 * e^(2x) * sin(x) step by step with the product rule, simplify the result, then evaluate "
                "f'(x) at x = 0 and at x = pi / 2. Write every formula out.",
                kMathTokens},
    ShortPrompt{"math-dice", "math",
                "Three fair six-sided dice are rolled. Work out, showing every step and every count, the probability that their sum "
                "is 10, and then the expected value of the largest of the three dice.",
                kMathTokens},
    ShortPrompt{"math-elimination", "math",
                "Solve the system 2x + 3y - z = 5, x - y + 4z = 6, 3x + 2y + 2z = 13 by Gaussian elimination. Show the augmented "
                "matrix after every row operation, give x, y and z as fractions, and verify them in all three equations.",
                kMathTokens},

    ShortPrompt{"prose-story", "prose",
                "Write a short story of about 600 words about a night-shift ferry pilot who notices that one passenger boards every "
                "night but never gets off.",
                kProseTokens},
    ShortPrompt{"prose-essay", "prose", "Write an essay of about 600 words on why cities built before the car are pleasant to walk in, with concrete examples.",
                kProseTokens},
    ShortPrompt{"prose-letter", "prose",
                "Write a warm, specific letter of about 500 words from a retiring physics teacher to the class she taught for the last time.", kProseTokens},

    ShortPrompt{"json-extract", "json",
                "Extract the people, places and dates from this text as JSON with the keys \"people\", \"places\" and \"dates\" (arrays "
                "of strings), and output only the JSON.\n\nOn 14 March 1879 Albert Einstein was born in Ulm; the family moved to "
                "Munich a year later. In 1905, while working at the patent office in Bern, he published four papers. Mileva Maric, "
                "whom he had met in Zurich, was his first wife. He received the Nobel Prize in 1921 and moved to Princeton in 1933.",
                kJsonTokens},
    ShortPrompt{"json-catalog", "json",
                "Produce a JSON array of eight fictional products for a hardware store. Each object has \"sku\" (string), \"name\", "
                "\"category\", \"price_eur\" (number), \"in_stock\" (boolean) and \"tags\" (array of strings). Output only the JSON.",
                kJsonTokens},
    ShortPrompt{"json-config", "json",
                "Write a JSON configuration for a web service with three environments (dev, staging, prod). Each has a database "
                "(host, port, pool size), a cache (ttl seconds, max entries), logging (level, format) and feature flags. Output only the JSON.",
                kJsonTokens},

    ShortPrompt{"edit-refactor", "edit",
                "Refactor this function so that it has no nested conditionals and no duplicated work, keep its behaviour, and return "
                "the whole new function.\n\n```python\ndef price(order):\n    total = 0\n    for item in order['items']:\n        if item['qty'] > 0:\n"
                "            if item['type'] == 'book':\n                total = total + item['qty'] * item['price'] * 0.93\n            else:\n"
                "                if item['type'] == 'food':\n                    total = total + item['qty'] * item['price'] * 0.93\n                else:\n"
                "                    total = total + item['qty'] * item['price'] * 0.81\n    if order['country'] == 'DE':\n        if total > 100:\n"
                "            total = total - 5\n    return round(total, 2)\n```",
                kEditTokens},
    ShortPrompt{"edit-bugfix", "edit",
                "This function should return the index of the first element that is >= target in a sorted list, or len(xs) if there is "
                "none. It is wrong for some inputs. Fix it, return the whole function, then name the inputs that failed.\n\n```python\n"
                "def lower_bound(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
                "        if xs[mid] < target:\n            lo = mid\n        else:\n            hi = mid - 1\n    return lo\n```",
                kEditTokens},
    ShortPrompt{"edit-add-feature", "edit",
                "Add an optional timeout in seconds to this function (None = wait forever) and raise TimeoutError when it expires. "
                "Return the whole new function.\n\n```python\nimport time\n\ndef wait_for(predicate, interval=0.1):\n    while True:\n"
                "        if predicate():\n            return True\n        time.sleep(interval)\n```",
                kEditTokens},

    ShortPrompt{"summary-incident", "summary",
                "Summarise this incident report in four sentences for a manager.\n\nAt 02:14 the primary database of the booking "
                "service stopped accepting writes after its disk filled up: a nightly export job had been changed the day before to "
                "write uncompressed files to the same volume. The on-call engineer was paged at 02:19, found the volume at 100 % at "
                "02:31 and deleted the export files at 02:38; writes resumed at 02:40. During the 26 minutes about 1,900 bookings "
                "failed and customers saw an error page. Reads were not affected. The export job has been moved to its own volume, an "
                "alert now fires at 80 % disk usage, and the change that removed the compression was not reviewed, which the team "
                "will address with a rule that job changes need a second pair of eyes.",
                kSummaryTokens},
    ShortPrompt{"summary-study", "summary",
                "Summarise this abstract in three sentences for a general reader.\n\nWe followed 4,200 adults aged 40 to 65 for nine "
                "years and recorded their daily step counts with wrist-worn devices for one week each year. Participants who "
                "averaged at least 7,000 steps a day had a 31 % lower rate of cardiovascular events than those under 4,000, after "
                "adjusting for age, sex, smoking, body mass index and income. The benefit levelled off above roughly 10,000 steps, "
                "and walking pace added no further effect once the step count was taken into account. Because step counts were "
                "measured for one week a year, and healthier people may simply walk more, the study shows an association and cannot "
                "prove that walking caused the difference.",
                kSummaryTokens},
    ShortPrompt{"summary-meeting", "summary",
                "Turn these meeting notes into a summary with the decisions and the open questions.\n\nNotes: release moved from "
                "the 12th to the 19th because the payment provider's sandbox was down for three days. Anna thinks the 19th is safe "
                "if the load test passes on Thursday; Ben is less sure because the new cache has only run in staging for a week. "
                "Agreed: the load test decides. If p99 stays under 300 ms we ship the 19th, otherwise the 26th. Nobody owns the "
                "customer email yet. Chloe will ask legal whether the new terms need a notice period. The mobile team needs the "
                "API freeze by Monday and has not heard back about the two new fields.",
                kSummaryTokens},
};

// The chats of the tasks on given input: the turns before each chat prompt's own.
constexpr std::array kChatTripDayEarlier = {
    EarlierTurn{"user", "I am visiting Porto for four days in May. I like seafood, viewpoints and long walks, not museums."},
    EarlierTurn{"assistant", "Porto in May suits that well: mild weather, long evenings and plenty of walking between the river, the old town and the coast. Shall I plan it day by day?"},
};
constexpr std::array kChatLaptopEarlier = {
    EarlierTurn{"user", "I need a laptop for university, mostly writing essays and some Python."},
    EarlierTurn{"assistant", "Then a light machine with a good keyboard, 16 GB of memory and a long battery life matters more than a strong graphics card. What is your budget?"},
};
constexpr std::array kChatLasagneEarlier = {
    EarlierTurn{"user", "I am cooking lasagne for six people tonight."},
    EarlierTurn{"assistant", "Nice. A classic one with a meat ragu and a bechamel takes about two and a half hours, most of it simmering. Do you want the recipe?"},
};
constexpr std::array kChatGitEarlier = {
    EarlierTurn{"user", "I committed to main by mistake instead of to a feature branch."},
    EarlierTurn{"assistant", "That is easy to fix as long as the commit has not been pushed. Has it?"},
};

// The tasks on given input: four prompts per category in a row. Each reply mostly repeats or reshapes what its prompt
// gives, and each max_tokens sits a little above the reply the task asks for, so replies end where the model ends them.
constexpr std::array kGivenPrompts = {
    ShortPrompt{"chat-trip-day", "chat",
                "Yes, but only day 3 for now: morning to evening, with places to eat.",
                350, kChatTripDayEarlier},
    ShortPrompt{"chat-laptop", "chat",
                "About 900 euros. Give me three options as a short list, one line each.",
                200, kChatLaptopEarlier},
    ShortPrompt{"chat-lasagne", "chat",
                "One guest is vegetarian. What do I change, step by step?",
                300, kChatLasagneEarlier},
    ShortPrompt{"chat-git", "chat",
                "No, it has not been pushed. Show me the exact commands.",
                250, kChatGitEarlier},
    ShortPrompt{"code-fix-average", "code",
                "This function should return the average of the positive numbers in a list, or 0.0 when there are none. It has two bugs. Return the corrected function, then one line per fix.\n"
                "\n"
                "```python\n"
                "def average_positive(values):\n"
                "    total = 0\n"
                "    count = 0\n"
                "    for v in values:\n"
                "        if v >= 0:\n"
                "            total += v\n"
                "        count += 1\n"
                "    return total / count\n"
                "```",
                300},
    ShortPrompt{"code-dataclass", "code",
                "Turn this dictionary-based code into a Python dataclass `Book` with the same fields and a method `is_long()` that returns True above 400 pages, and change both functions to use it. Return the whole module.\n"
                "\n"
                "```python\n"
                "def make_book(title, author, pages, year):\n"
                "    return {\"title\": title, \"author\": author, \"pages\": pages, \"year\": year}\n"
                "\n"
                "\n"
                "def describe(book):\n"
                "    return f\"{book['title']} by {book['author']} ({book['year']}), {book['pages']} pages\"\n"
                "```",
                450},
    ShortPrompt{"code-sql-report", "code",
                "Tables: employees(id, name, department_id, salary, hired_at) and departments(id, name). Write one SQL query that lists every department with its number of employees, its average salary rounded to two decimals and its most recent hire date, sorted by average salary, highest first.",
                250},
    ShortPrompt{"code-js-to-ts", "code",
                "Convert this JavaScript module to TypeScript: an interface for the cart item, explicit types on every parameter and return value. Return the whole file.\n"
                "\n"
                "```js\n"
                "export function addItem(cart, item) {\n"
                "  const found = cart.find((i) => i.sku === item.sku);\n"
                "  if (found) {\n"
                "    found.quantity += item.quantity;\n"
                "    return cart;\n"
                "  }\n"
                "  return [...cart, item];\n"
                "}\n"
                "\n"
                "export function removeItem(cart, sku) {\n"
                "  return cart.filter((i) => i.sku !== sku);\n"
                "}\n"
                "\n"
                "export function total(cart) {\n"
                "  return cart.reduce((sum, i) => sum + i.price * i.quantity, 0);\n"
                "}\n"
                "```",
                500},
    ShortPrompt{"reason-coins", "reasoning",
                "Nine coins look the same, but one of them is lighter than the others. With a balance scale, how do you find the light coin in two weighings? Explain each step.",
                400},
    ShortPrompt{"reason-who-where", "reasoning",
                "Anna, Ben and Chloe each work at one of three companies (Alpha, Beta, Gamma) and each lives in a different city (Berlin, Paris, Rome). Anna does not work at Beta. The person at Gamma lives in Rome. Ben lives in Paris. Chloe does not live in Rome. Ben does not work at Alpha. Who works where and lives where? Show the reasoning.",
                450},
    ShortPrompt{"reason-cache", "reasoning",
                "A web page takes 800 ms to load: 500 ms waiting for the database, 200 ms rendering and 100 ms on the network. A cache would cut the database time by 80 % for 60 % of the requests. What is the new average load time? Explain step by step.",
                350},
    ShortPrompt{"reason-syllogism", "reasoning",
                "Every Glimp is a Torv, and some Torvs are blue. Must some Glimps be blue? Answer yes or no, then explain in two sentences.",
                150},
    ShortPrompt{"math-discount", "math",
                "A jacket costs 120 euros. Its price is cut by 25 %, then the reduced price is raised by 10 %. What is the final price, and what single percentage change gives the same result? Show each step.",
                300},
    ShortPrompt{"math-trains", "math",
                "Two trains leave stations 300 km apart at the same moment and travel towards each other at 80 km/h and 70 km/h. When do they meet, and how far from the first station? Show the work.",
                300},
    ShortPrompt{"math-multiples", "math",
                "Find the sum of all integers from 1 to 200 that are divisible by 3 or by 5. Show the method.",
                400},
    ShortPrompt{"math-rectangle", "math",
                "A rectangle is 3 cm longer than twice its width, and its area is 65 square centimetres. Find both sides, showing the equation and how you solve it.",
                350},
    ShortPrompt{"prose-continue", "prose",
                "Continue this opening in the same voice for about 300 words:\n"
                "\n"
                "The bakery on Linden Street opened at five, and by ten past the queue already reached the corner. Nobody in it was talking; they were all watching the woman at the counter, who had asked for something the baker had never heard of.",
                450},
    ShortPrompt{"prose-plain", "prose",
                "Rewrite this paragraph in plain, friendly English for a newsletter, keeping every fact:\n"
                "\n"
                "\"Pursuant to the decision of the board, the community garden shall henceforth be accessible to members between the hours of 07:00 and 20:00, with the exception of Mondays, on which day maintenance activities shall be undertaken.\"",
                200},
    ShortPrompt{"prose-product", "prose",
                "Write a product description of about 200 words for a stainless steel water bottle that keeps drinks cold for 24 hours, holds 750 ml and has a leak-proof lid.",
                300},
    ShortPrompt{"prose-poem", "prose", "Write a ten-line poem about the first snow in a small town. No rhyme needed.", 150},
    ShortPrompt{"json-from-csv", "json",
                "Convert this CSV to a JSON array of objects, numbers as numbers and booleans as booleans. Return only the JSON.\n"
                "\n"
                "city,country,population,capital\n"
                "Vienna,Austria,1982000,true\n"
                "Graz,Austria,295000,false\n"
                "Linz,Austria,210000,false\n"
                "Zurich,Switzerland,421000,false\n"
                "Bern,Switzerland,134000,true",
                300},
    ShortPrompt{"json-order", "json",
                "Extract the order from this email as a JSON object with the keys customer, items (an array of objects with product and quantity), delivery_date (ISO) and express (boolean). Return only the JSON.\n"
                "\n"
                "\"Hi, this is Marta Klein. Please send 3 boxes of A4 paper, 2 black toner cartridges and 1 desk lamp to our office by 14 March 2027. Express delivery please, we are running out.\"",
                200},
    ShortPrompt{"json-tool-call", "json",
                "Write one JSON object that calls the function book_meeting_room with the arguments room, date (ISO), start (HH:MM), minutes and attendees (an array) for: \"Book the Orion room for a 90-minute planning session with Lea, Tom and Sofia next Thursday at 10:30.\" Today is Monday 2027-01-11. Return only the JSON.",
                150},
    ShortPrompt{"json-from-yaml", "json",
                "Convert this YAML to JSON. Return only the JSON.\n"
                "\n"
                "service:\n"
                "  name: billing\n"
                "  replicas: 3\n"
                "  ports:\n"
                "    - 8080\n"
                "    - 9090\n"
                "  env:\n"
                "    LOG_LEVEL: info\n"
                "    DB_POOL: 20\n"
                "  healthcheck:\n"
                "    path: /health\n"
                "    interval_s: 15",
                200},
    ShortPrompt{"edit-diff-rename", "edit",
                "Rename the function `fetch` to `fetch_user`, add type hints, and return None when the id is negative. Return a unified diff only.\n"
                "\n"
                "```python\n"
                "import requests\n"
                "\n"
                "BASE = \"https://api.example.com\"\n"
                "\n"
                "\n"
                "def fetch(user_id):\n"
                "    resp = requests.get(f\"{BASE}/users/{user_id}\", timeout=10)\n"
                "    resp.raise_for_status()\n"
                "    return resp.json()\n"
                "```",
                300},
    ShortPrompt{"edit-search-replace", "edit",
                "Return only one SEARCH/REPLACE block that changes the retries from 3 to 5 and the backoff from 0.5 to 2.0 in this line:\n"
                "\n"
                "client = HttpClient(base_url=URL, retries=3, backoff=0.5, verify=True)",
                120},
    ShortPrompt{"edit-add-field", "edit",
                "Add a field `email: str` to the User class, include it in to_dict() and in from_dict() (an empty string when the key is missing). Return the whole updated file.\n"
                "\n"
                "```python\n"
                "from dataclasses import dataclass\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class User:\n"
                "    id: int\n"
                "    name: str\n"
                "    active: bool = True\n"
                "\n"
                "    def to_dict(self):\n"
                "        return {\"id\": self.id, \"name\": self.name, \"active\": self.active}\n"
                "\n"
                "    @staticmethod\n"
                "    def from_dict(d):\n"
                "        return User(id=d[\"id\"], name=d[\"name\"], active=d.get(\"active\", True))\n"
                "```",
                350},
    ShortPrompt{"edit-config", "edit",
                "In this configuration, move the db_ settings under a \"database\" key (without the prefix), replace \"debug\": true with \"log_level\": \"debug\", and keep everything else. Return the whole new file.\n"
                "\n"
                "```json\n"
                "{\n"
                "  \"debug\": true,\n"
                "  \"host\": \"0.0.0.0\",\n"
                "  \"port\": 8080,\n"
                "  \"db_host\": \"localhost\",\n"
                "  \"db_port\": 5432,\n"
                "  \"db_name\": \"shop\",\n"
                "  \"workers\": 4\n"
                "}\n"
                "```",
                250},
    ShortPrompt{"summary-bullets", "summary",
                "Summarise this in exactly three bullet points:\n"
                "\n"
                "The city council reviewed the bus network after a year of the new timetable. Ridership grew 9 %, mostly on the two cross-town lines that now run every ten minutes. Late-evening services were used less than planned, and the council will cut two of the four night routes in spring. Complaints about crowding at the main station rose, so an extra platform will open in the autumn.",
                150},
    ShortPrompt{"summary-one-line", "summary",
                "Give a one-sentence summary of this: A survey of 3,000 households found that families who planned their meals for the week threw away 40 % less food than those who did not and spent about 15 euros less per week on groceries, although the planning took them around an hour every Sunday.",
                80},
    ShortPrompt{"summary-headings", "summary",
                "Summarise this report under the headings Problem, Change, Result and Open questions.\n"
                "\n"
                "A warehouse found that 6 % of its orders shipped late, mostly because pickers walked long routes between distant shelves. It moved the 500 most-ordered products to the shelves nearest the packing area and gave pickers handheld scanners that suggest a route. Over three months late orders fell to 2 % and the average pick took 35 % less time. Staff report that the scanners lose their signal in the back aisles, and it is unclear whether the gains will hold in the busy season.",
                300},
    ShortPrompt{"summary-thread", "summary",
                "Summarise this email thread in four short lines for someone who was not on it.\n"
                "\n"
                "Mark: The supplier says the new screens arrive on the 20th, not the 6th.\n"
                "Julia: Then the shop refit cannot finish before the opening on the 15th.\n"
                "Mark: We could open with the old screens and swap them later.\n"
                "Julia: Fine, but marketing must drop the \"new look\" line from the flyers.\n"
                "Mark: I will tell them today and book the installers for the 21st.",
                150},
};

// The shares of a coding agent's generated tokens, from what an agent harness reading, editing, building and testing
// code generates, split by what was generated - the thinking
// (reasoning), the shell commands and scripts and source files written (code), the patches (edit), the other tool-call
// arguments and call envelopes (json), notes and docs written (prose), the narration beside tool calls (chat) and the
// report that ends a turn of work (summary). The weights below are these shares; maths is never asked for.
constexpr std::array kAgentMix = {
    CategoryWeight{"chat", 0.04}, CategoryWeight{"code", 0.19}, CategoryWeight{"reasoning", 0.63}, CategoryWeight{"math", 0.00},
    CategoryWeight{"prose", 0.03}, CategoryWeight{"json", 0.03}, CategoryWeight{"edit", 0.04}, CategoryWeight{"summary", 0.04},
};
// Every kind of output alike: the categories' own rates, no deployment in mind.
constexpr std::array kEvenMix = {
    CategoryWeight{"chat", 0.125}, CategoryWeight{"code", 0.125}, CategoryWeight{"reasoning", 0.125}, CategoryWeight{"math", 0.125},
    CategoryWeight{"prose", 0.125}, CategoryWeight{"json", 0.125}, CategoryWeight{"edit", 0.125}, CategoryWeight{"summary", 0.125},
};
// The benchmark mix (the default headline): the shares the usual public benchmarks weigh their categories by, led by
// code, chat and maths not counted, averaged as they average it (MixAverage::arithmetic), so the figure stands beside
// theirs. It reads higher than the agentic mix: a fast category lifts a plain mean more than a slow one pulls it down.
constexpr std::array kBenchmarkMix = {
    CategoryWeight{"chat", 0.00}, CategoryWeight{"code", 0.30}, CategoryWeight{"reasoning", 0.20}, CategoryWeight{"math", 0.00},
    CategoryWeight{"prose", 0.15}, CategoryWeight{"json", 0.15}, CategoryWeight{"edit", 0.10}, CategoryWeight{"summary", 0.10},
};
constexpr std::array kMixes = {
    WeightMix{"benchmark", "the usual public benchmarks' shares: 30 % code, 20 % reasoning, 15 % each prose, json, 10 % each edit, summary; the weighted mean of the categories' rates",
              kBenchmarkMix, MixAverage::arithmetic},
    WeightMix{"agentic", "a coding agent's generated tokens on Qwen3.8 27B and Flash-Next (measured: 63 % reasoning, 19 % code, 4 % each edit, summary, chat, 3 % each prose, json); the rate that mix of tokens comes out at",
              kAgentMix},
    WeightMix{"even", "every kind of output alike", kEvenMix},
};

}  // namespace

std::span<const ShortPrompt> short_prompts(PromptSet set) {
  switch (set) {
    case PromptSet::open: return kPrompts;
    case PromptSet::given: return kGivenPrompts;
  }
  return kPrompts;
}

std::span<const ShortPrompt> measured_prompts() {
  static const std::vector<ShortPrompt> all = [] {
    std::vector<ShortPrompt> out;
    for (const CategoryWeight &c : kAgentMix)  // per category in the usual order: its open tasks, then its tasks on given input
      for (const auto set : {std::span<const ShortPrompt>(kPrompts), std::span<const ShortPrompt>(kGivenPrompts)})
        for (const ShortPrompt &p : set)
          if (p.category == c.category) out.push_back(p);
    return out;
  }();
  return all;
}
std::span<const WeightMix> weight_mixes() { return kMixes; }
std::span<const CategoryWeight> category_weights() { return kAgentMix; }
std::span<const WeightMix> headline_mixes() { return std::span(kMixes).first(2); }

}  // namespace xenithium:bench
