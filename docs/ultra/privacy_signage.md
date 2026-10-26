# Privacy and signage note for store operators

**This is general guidance written by the BREE team. It is not legal advice.** Rules about cameras, video analytics,
signage and retention differ by country, state and city, and by whether the cameras cover staff or customers. Ask a
lawyer, or your state or local authority, before you rely on any sentence below. We make no legal claim.

## What BREE does with your cameras, in plain words

* It watches the video your cameras already record, on a small computer, to notice things like someone leaving with an item
  that was not on a receipt, how busy the store is, and how long the line is.
* It does **not** keep pictures of faces, does not try to find out who anyone is, and does not link a person from one visit to
  the next.
* When it raises a possible theft it saves a short clip for you to review. In that clip heads are pixelated by the software.
  Everything else is deleted automatically after the retention period below.
* Your staff are in the picture too. BREE does not rate or track employees' performance. Tell staff the cameras and the
  software exist, in writing, before you switch it on.

## What you should do

1. **Tell people.** Put a clear sign at each entrance and, where it applies, near the register and pumps. Suggested text
   (adapt, and have it checked locally):
   > "This store uses video cameras and automated software to help prevent theft and to understand store traffic. Faces are
   > pixelated in saved clips. Clips are kept for 30 days unless needed to report a crime. Questions: [name, phone]."
   Use a symbol people recognise and make the text readable from the door.
2. **Written agreement for the pilot.** Get the owner's written OK, and say who can view clips (a short named list).
3. **Know where video goes.** If your DVR is reached through a cloud service (the Imou pilot is), the video passes through
   that company's servers. Say so in your own privacy notice. A direct local connection avoids this.
4. **Keep it short.** Default retention in BREE: pixelated alert clips 30 days, debug video 7 days, raw unpixelated segments
   24 hours (and only if you turn raw recording on, which we advise against). Shorter is better. Keep a clip longer only
   when you are handing it to police or an insurer, and mark it as a hold.
5. **Limit access.** Review clips only on the store's own computer or a protected account. Every clip view and every deletion
   is written to an audit log you can show on request.
6. **Requests.** If a customer or employee asks about footage of them, write down the request and answer within the time your
   local rules require. Pixelated clips usually cannot be matched to a person; the raw recording (if kept) can.
7. **No audio.** BREE does not record sound. Do not add microphones: recording conversations is regulated more strictly.
8. **Do not use it for** pointing at a named person, matching faces to a watch list, or any decision about an employee
   without a human looking at the evidence.

## What the pixelation does and does not do

Heads and faces are blocked into a coarse grid. Body shape, clothes, gait and the setting remain visible, and a regular
customer could be recognised by someone who knows them. The software can miss a face (see `trust.md` for measured misses on real
footage). Treat pixelation as a strong safeguard, not a guarantee.

## For us, the vendor

* Footage from a pilot store never leaves the machine it was recorded on and is not sent to any outside service.
* Labelling or demo material built from real footage must be pixelated first (`python -m bree.trust redact`).
