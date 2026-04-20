export interface PhraseData {
  id: string;
  label: string;
  category: 
    | "🏠 Basic Needs & Daily Life" 
    | "👋 Greetings & Introductions" 
    | "📞 Communication" 
    | "🏥 Emergency & Medical" 
    | "🏫 School / Work / Social";
  description: string;
  /** Static image or SVG shown as thumbnail. Every phrase gets its own unique asset. */
  imageUrl: string;
  /** Optional video or animated GIF URL for motion demonstration. */
  videoUrl?: string;
  /** Two-handed sign — user must show both hands */
  twoHanded?: boolean;
  instructions: string;
}

export const PHRASES: PhraseData[] = [

  // ── 🏠 BASIC NEEDS & DAILY LIFE ──────────────────────────────────────────
  {
    id: "water",
    label: "I need water",
    category: "🏠 Basic Needs & Daily Life",
    description: "Request for water.",
    imageUrl: "/signs/phrases/water.png",
    instructions: "Form the 'W' hand (index, middle, and ring fingers extended and slightly spread) and tap your chin twice.",
  },
  {
    id: "food",
    label: "I need food",
    category: "🏠 Basic Needs & Daily Life",
    description: "Request for food.",
    imageUrl: "/signs/phrases/food_v2.png",
    instructions: "Bring your bunched fingertips to your mouth as if putting food in it.",
  },
  {
    id: "hungry",
    label: "I am hungry",
    category: "🏠 Basic Needs & Daily Life",
    description: "Express hunger.",
    imageUrl: "/signs/phrases/hungry.png",
    instructions: "Form a wide 'C' shape with one hand and move it down your chest from throat to stomach.",
  },
  {
    id: "thirsty",
    label: "I am thirsty",
    category: "🏠 Basic Needs & Daily Life",
    description: "Express thirst.",
    imageUrl: "/signs/phrases/thirsty.png",
    instructions: "Run your index finger down your throat.",
  },
  {
    id: "help_me",
    label: "I need help",
    category: "🏠 Basic Needs & Daily Life",
    description: "Request for general assistance.",
    imageUrl: "/signs/phrases/help.png",
    instructions: "Place a 'thumbs up' hand on your flat open palm and lift them together.",
  },
  {
    id: "tired",
    label: "I am tired",
    category: "🏠 Basic Needs & Daily Life",
    description: "Express fatigue.",
    imageUrl: "/signs/phrases/tired.png",
    instructions: "Place both hands flat on your chest and let them droop or slide downward to show exhaustion.",
  },
  {
    id: "rest",
    label: "I need rest / I want to sleep",
    category: "🏠 Basic Needs & Daily Life",
    description: "Request for sleep or rest.",
    imageUrl: "/signs/phrases/rest.png",
    instructions: "Place your palms together near your ear and tilt your head slightly as if sleeping on a pillow.",
  },
  {
    id: "medicine",
    label: "I need medicine",
    category: "🏠 Basic Needs & Daily Life",
    description: "Request for medicine.",
    imageUrl: "/signs/phrases/medicine_v2.png",
    instructions: "Place your dominant middle finger on your other palm and twist it as if grinding a pill.",
  },
  {
    id: "toilet",
    label: "I need the toilet / bathroom",
    category: "🏠 Basic Needs & Daily Life",
    description: "Request for the restroom.",
    imageUrl: "/signs/phrases/toilet_v2.png",
    instructions: "Make a T-shape with your hand (thumb between index and middle) and shake it slightly.",
  },
  {
    id: "cold",
    label: "I am cold / I need a blanket",
    category: "🏠 Basic Needs & Daily Life",
    description: "Express feeling cold.",
    imageUrl: "/signs/phrases/cold.png",
    instructions: "Make fists and bring them close to your chest while shivering your body.",
  },

  // ── 👋 GREETINGS & INTRODUCTIONS ─────────────────────────────────────────
  {
    id: "hello",
    label: "Hello / Namaste",
    category: "👋 Greetings & Introductions",
    description: "Common greeting.",
    imageUrl: "/signs/phrases/isl_hello_demo.png", // Pointing to the new premium asset
    instructions: "Hold your open hand up with fingers spread and wave slightly side to side, or bring palms together in front of chest.",
  },
  {
    id: "good_morning",
    label: "Good morning",
    category: "👋 Greetings & Introductions",
    description: "Morning greeting.",
    imageUrl: "/signs/phrases/morning.png",
    instructions: "Sweep your flat hand upward and outward like the sun rising.",
  },
  {
    id: "good_night_greet",
    label: "Good night",
    category: "👋 Greetings & Introductions",
    description: "Night greeting.",
    imageUrl: "/signs/phrases/learn.svg", // Placeholder
    instructions: "Bring one hand over the other and let them droop downward like sundown.",
  },
  {
    id: "name_is",
    label: "My name is ___",
    category: "👋 Greetings & Introductions",
    description: "Introducing oneself.",
    imageUrl: "/signs/phrases/name.svg",
    instructions: "Point to yourself, then tap the index and middle fingers of both hands together twice.",
  },
  {
    id: "meet_you",
    label: "Nice to meet you",
    category: "👋 Greetings & Introductions",
    description: "Greeting someone new.",
    imageUrl: "/signs/phrases/friend.svg",
    instructions: "Slide your dominant flat palm across your other flat palm in a smooth motion.",
  },
  {
    id: "how_are_you_greet",
    label: "How are you?",
    category: "👋 Greetings & Introductions",
    description: "Asking someone's well-being.",
    imageUrl: "/signs/phrases/how_are_you.png",
    instructions: "Form 'C' shapes with both hands and roll them outward from your chest.",
  },
  {
    id: "im_fine",
    label: "I am fine",
    category: "👋 Greetings & Introductions",
    description: "Responding positively.",
    imageUrl: "/signs/phrases/help.png", // Reuse thumbs up
    instructions: "Give a clean 'thumbs up' sign moving slightly forward.",
  },
  {
    id: "goodbye_greet",
    label: "Goodbye / See you later",
    category: "👋 Greetings & Introductions",
    description: "Farewell greeting.",
    imageUrl: "/signs/phrases/morning.png", // Placeholder
    instructions: "Hold your hand up and wave fingers back and forth.",
  },
  {
    id: "welcome",
    label: "Welcome",
    category: "👋 Greetings & Introductions",
    description: "Welcoming someone.",
    imageUrl: "/signs/phrases/come.svg",
    instructions: "Hold your flat hand palm up and bring it toward your body.",
  },
  {
    id: "thank_you",
    label: "Thank you",
    category: "👋 Greetings & Introductions",
    description: "Expressing gratitude.",
    imageUrl: "/signs/phrases/thank_you.png",
    instructions: "Touch your chin with your flat hand and move it forward toward the other person.",
  },

  // ── 📞 COMMUNICATION ─────────────────────────────────────────────────────
  {
    id: "call_me",
    label: "Call me",
    category: "📞 Communication",
    description: "Request for a phone call.",
    imageUrl: "/signs/phrases/phone.png",
    instructions: "Extend your thumb and pinky (as if a phone) and hold it near your ear.",
  },
  {
    id: "call_doctor_request",
    label: "Please call the doctor",
    category: "📞 Communication",
    description: "Request for medical help.",
    imageUrl: "/signs/phrases/phone.png", // Placeholder
    instructions: "Make the 'Phone' sign followed by the 'Doctor' sign (tapping wrist).",
  },
  {
    id: "call_family",
    label: "Please call my family",
    category: "📞 Communication",
    description: "Request to contact family.",
    imageUrl: "/signs/phrases/phone.png", // Placeholder
    instructions: "Make the 'Phone' sign followed by circling your hand near your heart.",
  },
  {
    id: "no_hear",
    label: "I cannot hear you",
    category: "📞 Communication",
    description: "Indicating hearing difficulty.",
    imageUrl: "/signs/phrases/no_hear.png",
    instructions: "Point to your ear followed by a horizontal 'No' wave of your hand.",
  },
  {
    id: "repeat_please",
    label: "Please repeat that",
    category: "📞 Communication",
    description: "Asking for repetition.",
    imageUrl: "/signs/phrases/come.svg", // Placeholder
    instructions: "Point your index finger and move it in a small circular motion toward yourself.",
  },
  {
    id: "speak_slowly",
    label: "Please speak slowly",
    category: "📞 Communication",
    description: "Asking for slower speech.",
    imageUrl: "/signs/phrases/speak_slowly.png",
    instructions: "Place one flat hand over the other and move them slowly apart horizontally.",
  },
  {
    id: "understand",
    label: "I understand",
    category: "📞 Communication",
    description: "Confirming understanding.",
    imageUrl: "/signs/phrases/understand.svg",
    instructions: "Hold your fist near your forehead and flick your index finger up quickly.",
  },
  {
    id: "no_understand",
    label: "I do not understand",
    category: "📞 Communication",
    description: "Indicating lack of understanding.",
    imageUrl: "/signs/phrases/understand.svg",
    instructions: "Same as 'understand' but shake your head side-to-side simultaneously.",
  },
  {
    id: "write_down",
    label: "Please write it down",
    category: "📞 Communication",
    description: "Requesting written communication.",
    imageUrl: "/signs/phrases/pen.png",
    instructions: "Mime writing with a pen on your other flat palm.",
  },
  {
    id: "help_comm",
    label: "Can you help me communicate?",
    category: "📞 Communication",
    description: "Asking for communication assistance.",
    imageUrl: "/signs/phrases/help.png", // Placeholder
    instructions: "Make the 'Help' sign followed by pointing to your mouth and hands.",
  },

  // ── 🏥 EMERGENCY & MEDICAL ──────────────────────────────────────────────
  {
    id: "need_doctor",
    label: "I need a doctor",
    category: "🏥 Emergency & Medical",
    description: "Urgent medical request.",
    imageUrl: "/signs/phrases/hospital.svg", // Placeholder
    instructions: "Tap the fingertips of one hand on the wrist of your other arm (as if checking pulse).",
  },
  {
    id: "in_pain",
    label: "I am in pain",
    category: "🏥 Emergency & Medical",
    description: "Indicating physical pain.",
    imageUrl: "/signs/phrases/pain.svg",
    instructions: "Twist both index fingers toward each other several times near the area of pain.",
  },
  {
    id: "ambulance_call",
    label: "Call an ambulance",
    category: "🏥 Emergency & Medical",
    description: "Emergency transport request.",
    imageUrl: "/signs/phrases/phone.png", // Placeholder
    instructions: "Make the 'Phone' sign followed by circling a hand above your head mimicking a siren.",
  },
  {
    id: "i_have_fever",
    label: "I have a fever",
    category: "🏥 Emergency & Medical",
    description: "Indicating fever.",
    imageUrl: "/signs/phrases/fever.png",
    instructions: "Place the back of your hand against your forehead and look unwell.",
  },
  {
    id: "feel_dizzy",
    label: "I feel dizzy",
    category: "🏥 Emergency & Medical",
    description: "Indicating dizziness.",
    imageUrl: "/signs/phrases/lost.png", // Placeholder
    instructions: "Circle both hands around your head while moving your head slowly.",
  },
  {
    id: "need_wheelchair",
    label: "I need my wheelchair",
    category: "🏥 Emergency & Medical",
    description: "Request for mobility aid.",
    imageUrl: "/signs/phrases/lost.png", // Placeholder
    instructions: "Point both index fingers down and move them in large circles mimicking wheels.",
  },
  {
    id: "i_am_deaf",
    label: "I am deaf",
    category: "🏥 Emergency & Medical",
    description: "Indicating hearing loss.",
    imageUrl: "/signs/phrases/no_hear.png", // Placeholder
    instructions: "Touch your ear with your index finger, then touch the side of your mouth.",
  },
  {
    id: "use_isl",
    label: "I use sign language",
    category: "🏥 Emergency & Medical",
    description: "Indicating communication method.",
    imageUrl: "/signs/phrases/understand.svg", // Placeholder
    instructions: "Rotate both index fingers around each other in circles in front of your chest.",
  },
  {
    id: "be_patient",
    label: "Please be patient with me",
    category: "🏥 Emergency & Medical",
    description: "Asking for patience.",
    imageUrl: "/signs/phrases/rest.png", // Placeholder
    instructions: "Put your thumb to your chin and drag it slowly down to your throat.",
  },
  {
    id: "emergency_urgent",
    label: "Emergency!",
    category: "🏥 Emergency & Medical",
    description: "Signal for help.",
    imageUrl: "/signs/phrases/emergency_v2.png",
    instructions: "Shake your fist rapidly side to side with an urgent expression.",
  },

  // ── 🏫 SCHOOL / WORK / SOCIAL ───────────────────────────────────────────
  {
    id: "go_school",
    label: "I am going to school",
    category: "🏫 School / Work / Social",
    description: "Stating destination.",
    imageUrl: "/signs/phrases/school_v2.png",
    instructions: "Clap your flat hands together twice horizontally (School sign) then point forward.",
  },
  {
    id: "more_time_req",
    label: "I need more time",
    category: "🏫 School / Work / Social",
    description: "Requesting extra time.",
    imageUrl: "/signs/phrases/when.svg", // Reuse watch tap
    instructions: "Tap your wrist, then bring your fingertips together to make the 'more' sign.",
  },
  {
    id: "have_question",
    label: "I have a question",
    category: "🏫 School / Work / Social",
    description: "Asking for permission to speak.",
    imageUrl: "/signs/phrases/what.svg", // Placeholder
    instructions: "Raise your index finger and move it in a small question mark shape in the air.",
  },
  {
    id: "can_explain",
    label: "Can you please explain?",
    category: "🏫 School / Work / Social",
    description: "Requesting clarification.",
    imageUrl: "/signs/phrases/what.svg", // Placeholder
    instructions: "Hold both hands with fingers spread and alternate them moving forward and back.",
  },
  {
    id: "i_agree",
    label: "I agree",
    category: "🏫 School / Work / Social",
    description: "Stating agreement.",
    imageUrl: "/signs/phrases/yes.png", // Reusing yes
    instructions: "Point to your head then bring both index fingers together horizontally in front of you.",
  },
  {
    id: "i_disagree",
    label: "I disagree",
    category: "🏫 School / Work / Social",
    description: "Stating disagreement.",
    imageUrl: "/signs/phrases/dont-know.png", // Placeholder
    instructions: "Point to your head then move both index fingers away from each other sharply.",
  },
  {
    id: "yes_simple",
    label: "Yes",
    category: "🏫 School / Work / Social",
    description: "Simple affirmation.",
    imageUrl: "/signs/phrases/yes.png",
    instructions: "Make a fist and nod it up and down like a head nodding yes.",
  },
  {
    id: "no_simple",
    label: "No",
    category: "🏫 School / Work / Social",
    description: "Simple negation.",
    imageUrl: "/signs/phrases/no.svg",
    instructions: "Bring your index and middle fingers down to meet your thumb in a snapping motion.",
  },
  {
    id: "maybe_not_sure",
    label: "Maybe / I am not sure",
    category: "🏫 School / Work / Social",
    description: "Expressing uncertainty.",
    imageUrl: "/signs/phrases/lost.png", // Placeholder
    instructions: "Hold both open palms up and alternate them going up and down slightly.",
  },
  {
    id: "please_wait_now",
    label: "Please wait",
    category: "🏫 School / Work / Social",
    description: "Asking for a moment.",
    imageUrl: "/signs/phrases/wait.png",
    instructions: "Hold your hands with palms up and wiggle your fingers while moving hands side to side.",
  },
];
